import pytest
import torch
from spikingbrain_cpu.protocol import HIDDEN_SHAPE, tensor_summary, validate_hidden

def test_hidden_validation_and_summary():
    value=torch.arange(3584,dtype=torch.float32).reshape(HIDDEN_SHAPE)
    validate_hidden(value)
    summary=tensor_summary(value)
    assert summary['shape']==[1,1,3584]
    assert summary['nbytes']==14336
    assert summary['sentinels'][-1]==3583.0

def test_hidden_validation_rejects_shape():
    with pytest.raises(ValueError): validate_hidden(torch.zeros(1,3584))

def test_hidden_validation_rejects_dtype_and_nonfinite():
    with pytest.raises(TypeError): validate_hidden(torch.zeros(HIDDEN_SHAPE,dtype=torch.float64))
    value=torch.zeros(HIDDEN_SHAPE); value[0,0,0]=float('nan')
    with pytest.raises(FloatingPointError): validate_hidden(value)
