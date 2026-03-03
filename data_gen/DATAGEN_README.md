18/18 completed. Pipeline is working. A few observations from the run:                                                                           
                                                                                                                                                 
  What worked well: All 18 records generated, good quality output. The --workers 3 flag mostly doesn't help throughput here — Ollama processes one 
  request at a time on the ANE regardless, so parallel workers just queue up. Sequential is actually cleaner.                                      
  
  Healthcare had 4 parse failures — the model occasionally inverts relation directions (e.g. Medication | PRESCRIBED | Patient instead of Patient |
   PRESCRIBED | Medication). Those get filtered out and retried automatically.                                                                     
                                                                                                                                                   
  Speed reality check for the full run:                                                                                                            
  - ~20-30s per example                                                                                                                            
  - 20 domains × 50 examples = 1000 examples ≈ 6-7 hours                                                                                           
                                                                                                                                                   
  That's an overnight run. You could start it now with:                                                                                            
                                                                                                                                                   
  cd data_gen                                                                                                                                      
  uv run python3 generate.py --model qwen2.5:7b --per-domain 50 --workers 1 --output er_training_data.jsonl                                        
  # or resume if interrupted:                                                                                                                      
  uv run python3 generate.py --model qwen2.5:7b --per-domain 50 --workers 1 --output er_training_data.jsonl --resume