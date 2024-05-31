## Inference

We recommend use vllm or TGI for faster inference.

Start your vllm svr with `start_svr_vllm.sh` or tgi svr with `start_svr_tgi.sh`

The necessary parameters for these scripts are 
- GPU, GPU index (indices) for inference
- PORT, port for your serving
- MODEL, model path, in huggingface format

### Run inference

vllm 
```shell
python run_agent_two_stage_0_act_baseline_dst_vllm.py \
  --dataset "" \
  --split "test" \
  --host "http://0.0.0.0:${PORT}"
```

tgi, the same with vllm call


## Metric

```shell
python metric.py --input_file ${YOU_INFERENCE_OUTPUT}
```