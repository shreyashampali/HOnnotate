#!/usr/bin/env bash
seq='0010'
python inference_seg.py --seq $seq
python inference_hand.py --seq $seq
python inference_obj.py --seq $seq

seq='0011'
python inference_seg.py --seq $seq
python inference_hand.py --seq $seq
python inference_obj.py --seq $seq

seq='0012'
python inference_seg.py --seq $seq
python inference_hand.py --seq $seq
python inference_obj.py --seq $seq