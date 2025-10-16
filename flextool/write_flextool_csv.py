import csv
import pandas as pd
from pathlib import Path

def write_unit_capacity(v, s, param, r, output_dir='output'):
    """Write unit capacity results to CSV"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    fn_unit_capacity = output_path / 'unit_capacity__period.csv'
    # Create MultiIndex for fast lookup
    if not s.ed_invest.empty:
        ed_invest_idx = pd.MultiIndex.from_frame(s.ed_invest[['entity', 'period']])
    else:
        ed_invest_idx = pd.MultiIndex.from_tuples([], names=['entity', 'period'])
    if not s.ed_divest.empty:
        ed_divest_idx = pd.MultiIndex.from_frame(s.ed_divest[['entity', 'period']])
    else:
        ed_divest_idx = pd.MultiIndex.from_tuples([], names=['entity', 'period'])
    
    with open(fn_unit_capacity, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['unit', 'period', 'existing', 'invested', 'divested', 'total'])
        
        periods = set(s.d_realized_period) | set(s.period_invest)
        
        for p in s.process_unit:
            for d in sorted(periods):
                existing = param.entity_all_existing.loc[d, p] if d in param.entity_all_existing.index and p in param.entity_all_existing.columns else 0
                
                invested = 0
                if (p, d) in ed_invest_idx:
                    invested = v.invest.loc[d, p] * param.entity_unitsize['value'][p] if d in v.invest.index and p in v.invest.columns else 0
                
                divested = 0
                if (p, d) in ed_divest_idx:
                    divested = v.divest.loc[d, p] * param.entity_unitsize['value'][p] if d in v.divest.index and p in v.divest.columns else 0
                
                total = r.entity_all_capacity.loc[d, p] if d in r.entity_all_capacity.index and p in r.entity_all_capacity.columns else 0
                
                writer.writerow([p, d, f'{existing:.8g}', f'{invested:.8g}', f'{divested:.8g}', f'{total:.8g}'])