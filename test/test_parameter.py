# import torch
# from caustic import EPL, Simulator, Sersic, FlatLambdaCDM


# For future PR currently this test fails
# def test_static_parameter_init():
    # module = EPL(FlatLambdaCDM(h0=0.7, Om0=0.3))
    # print(module.params)
    # module.to(dtype=torch.float16)
    # assert module.params.static.FlatLambdaCDM.h0.value.dtype == torch.float16
