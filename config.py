class CONFIG:
    sr = 32000
    hop_size = 320
    fmin = 50
    fmax = 16000
    mel_bins = 384
    window_size = 2048
    
    lr = 1e-3
    l2 = 1e-4
    p = 0.5
    
    period = 10
    skip = 5
    
    epochs = 50
    batch_size = 32 # 24 for b4, 32 for b0
    
    ####################
    
#     pre_k = 2
#     pre_batch_size = 32
#     pre_p = 0.5
#     pre_period = 10
#     pre_l2 = 1e-4
    
# 903 config
# class CONFIG:
#     sr = 32000
#     hop_size = 320
#     fmin = 50
#     fmax = 16000
#     mel_bins = 384
#     window_size = 2048
    
#     l2 = 1e-4
#     p = 0.5
    
#     period = 10
#     skip = 5
    
#     epochs = 50
#     batch_size = 32
    
#     valid_id = 0