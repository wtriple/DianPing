import torch
import torch.nn as nn
from opts import parse_opts

device = 'cuda:0'

# set parameters
window_time = 2  # 0.1/0.5/1/2 seconds
is_64 = True
is_se = True
is_se_band = is_se and True  # band attention
is_se_channel = is_se and True  # channel attention
wav_band = 1
eeg_band = 1   # 原来是5

max_epoch = 100
dataset_name = 'KUL'
data_special = 'END'
se_channel_type = 'avg'
se_band_type = 'max'

wav_channel = 1
eeg_channel = 64
eeg_channel_new = 64 if is_64 else 16
fs_data = 128
window_length = fs_data * window_time  # the length of each samples
vali_percent = 0.2
test_percent = 0.2
file_num_each_trail = 1  # the data format
cnn_ken_num = 16  # the number of the conv
fcn_input_num = cnn_ken_num

is_beyond_trail = False  # is cross trail
is_all_train = False  # is cross subject
isDS = False  # is use the wav data
channel_number = eeg_channel * eeg_band + 2 * wav_channel * wav_band

opts = parse_opts()
thresh = opts.thresh  # neuronal threshold
lens = opts.lens  # hyper-parameters of approximate function
decay = opts.decay  # decay constants
count_thresh = 0
s_lens = 8

class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(opts.thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - opts.thresh) < lens
        return grad_input * temp.float()


act_fun = ActFun.apply

def mem_update(x, mem, spike):
    mem = mem * decay * (1. - spike) + x
    spike = act_fun(mem)  # act_fun : approximation firing function
    return mem, spike

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze操作
        y = self.fc(y).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x) # 注意力作用每一个通道上

class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()

        self.se_channel = SE_Block(eeg_channel_new, 16)

        self.cnn_conv = nn.Sequential(
            nn.Conv2d(eeg_band, cnn_ken_num, (eeg_channel_new+1, 9), stride=(eeg_channel_new+1, 1)),
            nn.ReLU(),
        )

        self.cnn_conv_eeg1 = nn.Sequential(
            nn.Conv2d(eeg_band, cnn_ken_num, (eeg_channel_new+1, 9), stride=(eeg_channel_new+1, 1)),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1 * window_time)),
        )

        self.cnn_conv_eeg2 = nn.Sequential(
            nn.Conv2d(eeg_band, cnn_ken_num, (eeg_channel_new+1, 9), stride=(eeg_channel_new+1, 1)),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1 * window_time)),
        )

        self.cnn_fcn = nn.Sequential(
            nn.Linear(fcn_input_num * window_time*2, 2),
            # nn.Sigmoid(),
            # nn.Dropout(0.5),
            # nn.Linear(fcn_input_num, 2),
            nn.Softmax(dim=1),
        )

        self.conv_thresh = nn.Sequential(
            nn.Conv2d(eeg_band, 1, (eeg_channel_new, 9), stride=(eeg_channel_new, 1)),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
        )
        self.cfc =  nn.Conv1d(eeg_band, eeg_band, kernel_size=2, bias=False, groups=eeg_band)
        self.bn= nn.BatchNorm1d(eeg_band)

        self.count_thresh = count_thresh
        self.count = 0

    def forward(self, eeg, stimuli, epoch):
        # split the wav and eeg data
        batch_size = eeg.size(0)
        channel = eeg.size(1)
        time_window = eeg.size(3) // s_lens
        e_mem = e_spike = torch.zeros(batch_size, eeg_band, eeg_channel_new, s_lens, device=device)
        wav_mem1 = wav_spike1 = torch.zeros(batch_size, wav_band, wav_channel, s_lens, device=device)
        wav_mem2 = wav_spike2 = torch.zeros(batch_size, wav_band, wav_channel, s_lens, device=device)
        wav_a = stimuli[:, 0, :, :]
        wav_b = stimuli[:, 1, :, :]
        eeg = eeg.view(batch_size, eeg_band, eeg_channel_new, window_length)
        wav_a = wav_a.view(batch_size, wav_band, wav_channel, window_length)
        wav_b = wav_b.view(batch_size, wav_band, wav_channel, window_length)
        wav = torch.cat([wav_a, wav_b], dim=2)
        
        if epoch <=50:
            opts.thresh = 0.5
        elif 50 < epoch < 100:
            # Style pooling
            # AvgPool（全局平均池化）：
            mean = eeg.view(batch_size, channel, -1).mean(-1).unsqueeze(-1)
            # StdPool（全局标准池化）
            std = eeg.view(batch_size, channel, -1).std(-1).unsqueeze(-1)
            u = torch.cat((mean, std), -1)  # (b, c, 2)

            # Style integration
            # CFC（全连接层）
            z = self.cfc(u)  # (b, c, 1)
            # BN（归一化）
            z = self.bn(z)
            # Sigmoid
            g = torch.sigmoid(z)
            thresh = g.view(batch_size, channel, 1, 1)
            # print("thresh", thresh)
            opts.thresh = thresh.view(-1).mean()

            if 95<epoch<=100:
                self.count_thresh = self.count_thresh + opts.thresh
                self.count = self.count + 1 
                
        else:
            opts.thresh = self.count_thresh / self.count
        
                

        # SNN
        sum_spike = []
        sum_espike = []
        sum_wspike1 = []
        sum_wspike2 = []
        for step in range(time_window):
            eeg_step = eeg[:, :, :, step*s_lens:(step+1)*s_lens]
            wav_stepa = wav_a[:, :, :, step*s_lens:(step+1)*s_lens]
            wav_stepb = wav_b[:, :, :, step*s_lens:(step+1)*s_lens]
            e_mem, e_spike = mem_update(eeg_step, e_mem, e_spike)
            wav_mem1, wav_spike1 = mem_update(wav_stepa, wav_mem1, wav_spike1)
            wav_mem2, wav_spike2 = mem_update(wav_stepb, wav_mem2, wav_spike2)
            sum_espike.append(e_spike)
            sum_wspike1.append(wav_spike1)
            sum_wspike2.append(wav_spike2)
        sum_espike = torch.cat(sum_espike, dim=3)
        sum_wspike1 = torch.cat(sum_wspike1, dim=3)
        sum_wspike2 = torch.cat(sum_wspike2, dim=3)
        # print("sum_espike", sum_espike.shape)
        # sum_espike = self.pre_eeg(sum_espike)
        # sum_wspike = self.pre_stimuli(sum_wspike)
        
        
        # Channel attention
        if is_se_channel:
            sum_espike = sum_espike.view(batch_size, eeg_channel_new, 1, window_length)
            eeg = eeg.view(batch_size, eeg_channel_new, window_length)
            sum_espike = self.se_channel(sum_espike)

        sum_espike = sum_espike.view(batch_size, eeg_band, eeg_channel_new, window_length)
        
        # CNN
        #eeg_pre = self.cnn_conv(eeg)
        # print("sum_wspike1", sum_wspike1.shape)
        # print("sum_espike", sum_espike.shape)
        sum_spike1 = torch.cat([sum_wspike1, sum_espike], dim=2)
        sum_spike2 = torch.cat([sum_wspike2, sum_espike], dim=2)
    
        y1 = self.cnn_conv_eeg1(sum_spike1)
        y2 = self.cnn_conv_eeg2(sum_spike2)
        
        y1 = y1.view(batch_size, -1)
        y2 = y2.view(batch_size, -1)
        y = torch.cat([y1, y2], dim=1)
        # print("y", y.size())

        # convolution
        output = self.cnn_fcn(y)

        return output, opts.thresh