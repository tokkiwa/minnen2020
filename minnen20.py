from compressai.models import MeanScaleHyperprior
import torch
import torch.nn as nn
from compressai.layers import GDN
from compressai.models.utils import conv, deconv
from compressai.ans import RansEncoder, RansDecoder, BufferedRansEncoder, RansDecoder

class Minnen2020(MeanScaleHyperprior):
    '''
    Channel-wise Context Model proposed in David Minnen&Saurabh Singh, Channel-wise Autoregressive Entropy Models for Learned Image Compression. ICIP 2020. 
    *WITHOUT* LRP(Latent Residual Prediction) module. 
    '''
    def __init__(self, N=192, M=320, slice = 10, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv(M, M, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(M, (N + M)//2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv((N + M)//2, N, stride=2, kernel_size=5),
        ) 
        #320 -> 256 -> 192

        self.h_s = nn.Sequential(
            deconv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(N, (M + N) // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv((M + N) // 2, M, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
        )
        #192 -> 256 -> 320
        self.slice = slice 
        self.slice_size = M//self.slice #Channel size for one slice. Note that M % slice should be zero
        self.y_size_list = [(i + 1) * self.slice_size for i in range(self.slice -1)]
        self.y_size_list.append(M)    #[32, 64, 96, 128, 160, 192, 224, 256, 288, 320] if M = 320 and slice = 10
        EP_inputs = [i * self.slice_size for i in range(self.slice)]    #Input channel size for entropy parameters layer. [0, 32, 64, 96, 128, 160, 192, 224, 256, 288] if M = 320 and slice = 10
        self.EPlist = nn.ModuleList([])
        for y_size in EP_inputs:
            EP = nn.Sequential(
                conv(y_size + M, M - (N//2), stride=1, kernel_size= 3),
                nn.LeakyReLU(inplace=True),
                conv(M - (N//2), (M + N) // 4,  stride=1, kernel_size= 3),
                nn.LeakyReLU(inplace=True),
                conv((M + N) // 4, M * 2 // 10, stride=1, kernel_size=3),     
            )
            self.EPlist.append(EP)
        #Variable->224, 224->128, 128->32

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyper_params = self.h_s(z_hat)
        list_sliced_y = [] #Stores each slice of y
        for i in range(self.slice - 1):
            list_sliced_y.append(y[:,(self.slice_size * i):(self.slice_size * (i + 1)),:,:])
        list_sliced_y.append(y[:,self.slice_size * (self.slice - 1):,:,:])
        y_hat_cumul = torch.Tensor().to(y.device) #Cumulative y_hat. Stores already encoded y_hat slice
        scales_hat_list = []
        means_hat_list = []
        for i in range(self.slice):
            if i == 0:
                gaussian_params = self.EPlist[0](
                    hyper_params
                )
            else:
                gaussian_params = self.EPlist[i](
                    torch.cat([hyper_params, y_hat_cumul], dim = 1)
                )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            scales_hat_list.append(scales_hat)
            means_hat_list.append(means_hat)
            y_hat_sliced = self.gaussian_conditional.quantize(
               list_sliced_y[i] , "noise" if self.training else "dequantize"
            )
            y_hat_cumul = torch.cat([y_hat_cumul, y_hat_sliced], dim = 1)

        scales_all = torch.cat(scales_hat_list, dim = 1)
        means_all = torch.cat(means_hat_list, dim = 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_all, means=means_all)
        x_hat = self.g_s(y_hat_cumul)
    

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        encoder = BufferedRansEncoder()
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        indexes_list = []
        symbols_list = []
        y_strings = []        
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyper_params = self.h_s(z_hat)

        list_sliced_y = []
        for i in range(self.slice - 1):
            list_sliced_y.append(y[:,(self.slice_size * i):self.slice_size * (i + 1),:,:])
        list_sliced_y.append(y[:,self.slice_size * (self.slice - 1):,:,:])
        y_hat = torch.Tensor().to(x.device)
        for i in range(self.slice):
            y_sliced = list_sliced_y[i] #size[1, M/S * i, H', W']
            if i == 0 :
                gaussian_params = self.EPlist[0](
                    hyper_params
                )
            else: 
                gaussian_params = self.EPlist[i](
                    torch.cat([hyper_params, y_hat], dim=1)
                )
            #gaussian_params = gaussian_params.squeeze(3).squeeze(2) #size ([1,256])
            scales_hat, means_hat = gaussian_params.chunk(2, 1) 
            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            
            y_hat_sliced = self.gaussian_conditional.quantize(y_sliced, "symbols", means_hat)
            symbols_list.extend(y_hat_sliced.reshape(-1).tolist())
            indexes_list.extend(indexes.reshape(-1).tolist())
            y_hat_sliced = y_hat_sliced + means_hat
            
            y_hat = torch.cat([y_hat, y_hat_sliced], dim = 1)
           
        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        y_string = encoder.flush()
        y_strings.append(y_string)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}       

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        hyper_params = self.h_s(z_hat)
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        decoder = RansDecoder()
        decoder.set_stream(strings[0][0])

        y_hat = torch.Tensor().to(z_hat.device)
        for i in range(self.slice):
            if i == 0:
                gaussian_params = self.EPlist[0](hyper_params) 
            else:
                gaussian_params = self.EPlist[i](
                    torch.cat([hyper_params, y_hat], dim = 1)
                )
            scales_sliced, means_sliced = gaussian_params.chunk(2,1)
            indexes_sliced = self.gaussian_conditional.build_indexes(scales_sliced)
            y_sliced_hat = decoder.decode_stream(
                indexes_sliced.reshape(-1).tolist(), cdf, cdf_lengths, offsets
            )
            y_sliced_hat  =torch.Tensor(y_sliced_hat).reshape(scales_sliced.shape).to(scales_sliced.device)
            y_sliced_hat += means_sliced
            y_hat = torch.cat([y_hat, y_sliced_hat], dim = 1)
            
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}
    
class Minnen2020LRP(Minnen2020):
    '''
    Channel-wise Context Model proposed in David Minnen&Saurabh Singh, Channel-wise Autoregressive Entropy Models for Learned Image Compression. ICIP 2020. 
    *WITH* LRP(Latent Residual Prediction) module. 

    This model also separates scale parameter prediction network and mean parameter prediction network to represent the original paper.
    '''
    def __init__(self, N=192, M=320, slice = 10, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.slice = slice
        self.slice_size = M//self.slice
        y_size_list = [self.slice_size for _ in range(self.slice -1)]
        y_size_list.append(M - self.slice_size * (self.slice - 1))
        # [32, 32, 32, 32, 32, 32, 32, 32, 32, 32]
        y_inputs = [i * self.slice_size for i in range(self.slice)]
        # [0, 32, 64, 96, 128, 160, 192, 224, 256, 288]
        LRP_inputs = [(i+1) * self.slice_size for i in range(self.slice)]
        # [32, 64, 96, 128, 160, 192, 224, 256, 288, 320]
        self.LRPlist = nn.ModuleList([])
        self.scaleEPlist = nn.ModuleList([])
        self.meanEPlist = nn.ModuleList([])
        for y_cumul_size in y_inputs:
            scaleEP = nn.Sequential(
                conv(y_cumul_size + (M//2), M - (N//2), stride=1, kernel_size= 3),
                nn.ReLU(inplace=True),
                conv(M - (N//2), (M + N) // 4,  stride=1, kernel_size= 3),
                nn.ReLU(inplace=True),
                conv((M + N) // 4, M // self.slice, stride=1, kernel_size=3),     
            )
            self.scaleEPlist.append(scaleEP)
            meanEP = nn.Sequential(
                conv(y_cumul_size + (M//2), M - (N//2), stride=1, kernel_size= 3),
                nn.ReLU(inplace=True),
                conv(M - (N//2), (M + N) // 4,  stride=1, kernel_size= 3),
                nn.ReLU(inplace=True),
                conv((M + N) // 4, M // self.slice, stride=1, kernel_size=3),     
            )
            self.meanEPlist.append(meanEP)
        for y_cumul_size_alt in LRP_inputs:
            LRP = nn.Sequential(
                conv(y_cumul_size_alt + (M//2), M - (N//2), stride=1, kernel_size= 3),
                nn.ReLU(inplace=True),
                conv(M - (N//2), (M + N) // 4,  stride=1, kernel_size= 3),
                nn.ReLU(inplace=True),
                conv((M + N) // 4, M // self.slice, stride=1, kernel_size=3),     
            )
            self.LRPlist.append(LRP)
            #The in/out channel for LRP layers are designed to agree the values shown in paper when N = 192 and M = 320. 
    
    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyper_params = self.h_s(z_hat)
        hyper_scale, hyper_mean = hyper_params.chunk(2,1)
        list_sliced_y = []
        for i in range(self.slice - 1):
            list_sliced_y.append(y[:,(self.slice_size * i):self.slice_size * (i + 1),:,:])
        list_sliced_y.append(y[:,self.slice_size * (self.slice - 1):,:,:])
        y_hat = torch.Tensor().to(y.device)
        scales_hat_list = []
        means_hat_list = []
        for i in range(self.slice):
            if i == 0 :
                scales_hat = self.scaleEPlist[0](
                    hyper_scale
                )
                means_hat = self.meanEPlist[0](
                    hyper_mean
                )
            else: 
                scales_hat = self.scaleEPlist[i](
                    torch.cat([hyper_scale, y_hat], dim = 1)
                )
                means_hat = self.meanEPlist[i](
                    torch.cat([hyper_mean, y_hat], dim = 1)
                )
            scales_hat_list.append(scales_hat)
            means_hat_list.append(means_hat)
            y_hat_sliced = self.gaussian_conditional.quantize(
               list_sliced_y[i] , "noise" if self.training else "dequantize"
            ) 
            LRP_param = self.LRPlist[i](
                torch.cat([y_hat, y_hat_sliced, hyper_mean],dim = 1)
            )
            y_hat = torch.cat([y_hat, y_hat_sliced + LRP_param], dim = 1)

        
        scales_all = torch.cat(scales_hat_list, dim = 1)
        means_all = torch.cat(means_hat_list, dim = 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_all, means=means_all)
        x_hat = self.g_s(y_hat)
    

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        encoder = BufferedRansEncoder()
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        indexes_list = []
        symbols_list = []
        
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyper_params = self.h_s(z_hat)
        hyper_scale, hyper_mean = hyper_params.chunk(2,1)
        list_sliced_y = []
        for i in range(self.slice - 1):
            list_sliced_y.append(y[:,(self.slice_size * i):self.slice_size * (i + 1),:,:])
        list_sliced_y.append(y[:,self.slice_size * (self.slice - 1):,:,:])

        y_hat = torch.Tensor().to(y.device)
        for i in range(self.slice):
            y_sliced = list_sliced_y[i] #size[1, M/S * i, H', W']
            if i == 0 :
                scales_hat = self.scaleEPlist[0](
                    hyper_scale
                )
                means_hat = self.meanEPlist[0](
                    hyper_mean
                )
            else: 
                scales_hat = self.scaleEPlist[i](
                    torch.cat([hyper_scale, y_hat], dim = 1)
                )
                means_hat = self.meanEPlist[i](
                    torch.cat([hyper_mean, y_hat], dim = 1)
                )
            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            y_hat_sliced = self.gaussian_conditional.quantize(y_sliced, "symbols", means_hat)
            symbols_list.extend(y_hat_sliced.reshape(-1).tolist())
            indexes_list.extend(indexes.reshape(-1).tolist())
            y_hat_sliced = y_hat_sliced + means_hat

            #LRP configuration
            LRPparam = self.LRPlist[i](
                torch.cat([y_hat, y_hat_sliced, hyper_mean], dim = 1)
            )
            y_hat_sliced += LRPparam
            y_hat = torch.cat([y_hat, y_hat_sliced], dim = 1)
            
        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )
        y_strings = []
        y_string = encoder.flush()
        y_strings.append(y_string)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}       

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        hyper_params = self.h_s(z_hat)
        hyper_scale, hyper_mean = hyper_params.chunk(2,1)
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        decoder = RansDecoder()
        decoder.set_stream(strings[0][0])

        y_hat = torch.Tensor().to(z_hat.device)
        for i in range(self.slice):
            if i == 0 :
                scales_sliced = self.scaleEPlist[0](
                    hyper_scale
                )
                means_sliced = self.meanEPlist[0](
                    hyper_mean
                )
            else: 
                scales_sliced = self.scaleEPlist[i](
                    torch.cat([hyper_scale, y_hat], dim = 1)
                )
                means_sliced = self.meanEPlist[i](
                    torch.cat([hyper_mean, y_hat], dim = 1)
                )
            indexes_sliced = self.gaussian_conditional.build_indexes(scales_sliced)
            y_hat_sliced = decoder.decode_stream(
                indexes_sliced.reshape(-1).tolist(), cdf, cdf_lengths, offsets
            )
            y_hat_sliced  =torch.Tensor(y_hat_sliced).reshape(scales_sliced.shape).to(scales_sliced.device)
            y_hat_sliced += means_sliced
            LRPparam = self.LRPlist[i](
                torch.cat([y_hat,y_hat_sliced, hyper_mean], dim = 1)
            )
            
            y_hat_sliced += LRPparam
            y_hat = torch.cat([y_hat, y_hat_sliced], dim = 1)
            
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}
