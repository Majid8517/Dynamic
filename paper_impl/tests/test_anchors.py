import torch
from paper_impl.anchors import generate_anchors, encode_boxes, decode_boxes

def test_anchor_count_matches_position_major_head_order():
    anchors=generate_anchors((512,512),[(64,64),(32,32)],num_scales=3,aspect_ratios=(1.0,2.0,0.5),anchor_scale=4.0)
    assert anchors[0].shape==(64*64*9,4)
    assert anchors[1].shape==(32*32*9,4)

def test_box_encode_decode_roundtrip():
    anchors=torch.tensor([[0.,0.,10.,10.],[10.,10.,30.,30.]])
    gt=torch.tensor([[1.,2.,9.,8.],[12.,11.,28.,29.]])
    d=encode_boxes(gt,anchors)
    rec=decode_boxes(d,anchors)
    assert torch.allclose(rec,gt,atol=1e-5)
