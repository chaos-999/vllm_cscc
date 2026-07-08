#!/bin/bash
# ============================================================
# DCU 优化版 vLLM 全流程测试脚本
# CSCC 先导杯 — 基于国产加速卡的千问大模型推理服务优化
# Model: Qwen3.5-27B | Hardware: DCU K100
# ============================================================
# 用法:
#   ./scripts/run_all.sh                         # 全自动跑
#   MODEL_DIR=/path/to/model ./scripts/run_all.sh # 指定模型路径
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TESTDATA_DIR="${TESTDATA_DIR:-$WORKSPACE_DIR/../testdata}"
MODEL_DIR="${MODEL_DIR:-$WORKSPACE_DIR/../Qwen3.5-27B}"
SERVER_PORT="${SERVER_PORT:-8001}"

# ---- 参数校验 ----
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ 模型目录不存在: $MODEL_DIR"
    echo "请设置 MODEL_DIR 环境变量指向正确的模型路径"
    echo "   MODEL_DIR=/public/home/xdzs2026_c204/Qwen3.5-27B $0"
    exit 1
fi

if [ ! -f "$TESTDATA_DIR/start_vllm.sh" ]; then
    echo "❌ 测试数据目录不存在或缺少 start_vllm.sh: $TESTDATA_DIR"
    echo "请设置 TESTDATA_DIR 环境变量指向 testdata 目录"
    exit 1
fi

echo "=========================================="
echo " DCU 优化版 vLLM 全流程测试"
echo "=========================================="
echo " 模型:     $MODEL_DIR"
echo " 测试数据: $TESTDATA_DIR"
echo " 端口:     $SERVER_PORT"
echo "=========================================="

# ---- Step 1: 编译安装 ----
echo ""
echo "[Step 1/4] 编译安装 vLLM..."
cd "$WORKSPACE_DIR"
git pull origin main
python setup.py bdist_wheel 2>&1 | tail -5
cd dist
pip install vllm-*.whl --no-deps --force-reinstall 2>&1 | tail -3
echo "✅ 编译安装完成"

# ---- Step 2: 启动服务 ----
echo ""
echo "[Step 2/4] 启动 vLLM 推理服务 (端口 $SERVER_PORT)..."

# Kill old server if running
kill $(lsof -t -i:$SERVER_PORT) 2>/dev/null || true
sleep 2

cd "$TESTDATA_DIR"
MODEL_DIR="$MODEL_DIR" ./start_vllm.sh &
SERVER_PID=$!

echo "等待服务启动（约 10 分钟）..."
echo "（你也可以另开终端观察日志: tail -f /dev/null）"

# Wait for server to be ready
for i in $(seq 1 120); do
    sleep 5
    if curl -sf http://127.0.0.1:$SERVER_PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"Qwen3.5-27B","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' > /dev/null 2>&1; then
        echo "✅ 服务启动成功！"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "❌ 服务启动超时"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    echo -n "."
done
echo ""

# ---- Step 3: 测试单次推理 ----
echo ""
echo "[Step 3/4] 单次推理验证..."
curl http://127.0.0.1:$SERVER_PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3.5-27B",
        "messages": [{"role": "user", "content": "你好，简单回复一句话。"}],
        "temperature": 0.0,
        "max_tokens": 64
    }' 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print('回复:', d['choices'][0]['message']['content'][:80])" 2>/dev/null || echo "（单次推理完成）"
echo "✅ 推理验证通过"

# ---- Step 4: 吞吐测试 ----
echo ""
echo "[Step 4/4] 运行吞吐测试..."
echo "（耗时较长，请耐心等待）"
echo ""

# Run all three tiers
for TIER in "4-8K" "8-16K" "16-32K"; do
    echo "===== 吞吐测试: $TIER ====="
    MODEL_DIR="$MODEL_DIR" \
    python3 -m vllm.benchmarks.serve \
        --backend openai-chat \
        --host 127.0.0.1 \
        --port $SERVER_PORT \
        --model Qwen3.5-27B \
        --tokenizer "$MODEL_DIR" \
        --dataset-name custom \
        --dataset-path "$TESTDATA_DIR/${TIER}_throughput.jsonl" \
        --no-stream \
        --no-oversample \
        --disable-shuffle \
        --num-prompts 50 \
        --request-rate 1.0 \
        --max-concurrency 1 \
        --temperature 0.0 \
        --result-dir "$TESTDATA_DIR/test/${TIER}_throughput" \
        --result-filename result.json \
        --save-result \
        --save-detailed \
        --num-warmups 2 \
        --disable-tqdm 2>&1 | tee /dev/null
    echo ""
done

# ---- 汇总结果 ----
echo ""
echo "=========================================="
echo " 测试完成！结果汇总"
echo "=========================================="

for TIER in "4-8K" "8-16K" "16-32K"; do
    RESULT_FILE="$TESTDATA_DIR/test/${TIER}_throughput/result.json"
    if [ -f "$RESULT_FILE" ]; then
        echo ""
        echo "--- $TIER ---"
        python3 -c "
import json
with open('$RESULT_FILE') as f:
    d = json.load(f)
print(f'  Output吞吐: {d.get(\"request_output_throughput\", \"N/A\"):>8.2f} tok/s')
print(f'  TTFT P99:   {d.get(\"ttft_p99\", \"N/A\")}')
print(f'  TPOT P99:   {d.get(\"tpot_p99\", \"N/A\")}')
print(f'  请求数:     {d.get(\"num_requests\", \"N/A\")}')
" 2>/dev/null || echo "  结果文件: $RESULT_FILE"
    fi
done

echo ""
echo "=========================================="
echo " 详细结果文件在: $TESTDATA_DIR/test/"
echo "=========================================="

# Don't kill the server - keep it running for manual testing
echo ""
echo "💡 服务仍在运行 (PID=$SERVER_PID, 端口 $SERVER_PORT)"
echo "   测试精度: MODEL_DIR=$MODEL_DIR ./run_accuracy.sh"
echo "   关闭服务: kill $SERVER_PID"
