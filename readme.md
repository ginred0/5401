## Run

Run IQ:

```bash
python -m rec_simulator.IQ_recommender \
  --task movie \
  --method iq \
  --backbone deepseek-thinking-low \
  --max-users 200
```

Run all baselines:

```bash
for method in direct_llm pebol decisive gate; do
  python -m rec_simulator.IQ_recommender \
    --task iq \
    --method "$method" \
    --backbone deepseek-thinking-low \
    --max-users 200
done
```
