import csv
import pandas as pd
from pathlib import Path

def write_unit_capacity(v, s, param, r, output_dir='output'):
    """Write unit capacity results to CSV"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    fn_unit_capacity = output_path / 'unit_capacity__period.csv'
    # Create MultiIndex for fast lookup
   
    with open(fn_unit_capacity, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['unit', 'period', 'existing', 'invested', 'divested', 'total'])
        
        periods = set(s.d_realized_period) | set(s.period_invest)
        
        for p in s.process_unit:
            for d in periods:
                existing = param.entity_all_existing.droplevel(0, axis=1).loc[d, p]
                invested = 0
                if (p, d) in s.ed_invest:
                    invested = v.invest.loc[d, p] * param.entity_unitsize[p] if d in v.invest.index and p in v.invest.columns else 0
                divested = 0
                if (p, d) in s.ed_divest:
                    divested = v.divest.loc[d, p] * param.entity_unitsize[p] if d in v.divest.index and p in v.divest.columns else 0
                total = r.entity_all_capacity.droplevel(0, axis=1).loc[d, p]
                writer.writerow([p, d, f'{existing:.8g}', f'{invested:.8g}', f'{divested:.8g}', f'{total:.8g}'])