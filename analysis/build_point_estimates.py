from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    DATA_ROOT, RUN_META, load_run_metrics, metric_public_row, median, read_csv,
    sha256_file, write_csv,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'analysis'
MAN = ROOT / 'data' / 'manifests'
for p in (OUT, MAN):
    p.mkdir(parents=True, exist_ok=True)

RUNS = [
    'cfpb_online_recal1_seed0', 'cfpb_online_recal0_seed0',
    'cfpb_online_recal1_tiw010_seed0', 'cfpb_online_recal1_tiw020_seed0',
    'cfpb_recal_win36_seed0', 'cfpb_recal_eta015_seed0',
    'cfpb_recal_upd3_seed0', 'cfpb_recal_maxup110_seed0',
    'cfpb_gradual_bias_seed0',
    'smap_A1_weekly_ledger_seed0', 'smap_D15_weekly_ledger_seed0',
    'smap_A1_gradual_bias_seed0', 'smap_D15_gradual_bias_seed0',
]

all_metrics: dict[str, dict[tuple[str,str,str], dict[str,Any]]] = {}
all_public: list[dict[str,Any]] = []
run_manifests: dict[str,Any] = {}
for run in RUNS:
    metrics, manifest, args = load_run_metrics(run)
    all_metrics[run] = metrics
    run_manifests[run] = {'source_manifest': manifest, 'parsed_args': args, 'n_selected_rows': len(metrics)}
    all_public.extend(metric_public_row(run, m) for m in metrics.values())

write_csv(OUT/'point_estimates_all_selected_strength_B1.csv', all_public)

# Canonical CFPB structures.
on = all_metrics['cfpb_online_recal1_seed0']
off = all_metrics['cfpb_online_recal0_seed0']
attack_kinds = ('freeze','corr_mix')

def attacked(metrics, kind=None):
    vals=[]
    for key,m in metrics.items():
        k,p,d=key
        if p!='score_only': continue
        if kind is not None and k!=kind: continue
        vals.append(m)
    return vals

freeze_on = {m['spec'].detector:m for m in attacked(on,'freeze')}
corr_on = {m['spec'].detector:m for m in attacked(on,'corr_mix')}
freeze_off = {m['spec'].detector:m for m in attacked(off,'freeze')}
corr_off = {m['spec'].detector:m for m in attacked(off,'corr_mix')}
if set(freeze_on)!=set(corr_on) or len(freeze_on)!=19:
    raise RuntimeError('Expected the same 19 attacked detectors under Freeze and CorrMix.')

# Clean-workload invariance across incident families.
invariance=[]
for d in sorted(freeze_on):
    a,b=freeze_on[d],corr_on[d]
    same_ab=bool(np.array_equal(a['a_b'],b['a_b']))
    same_aj=bool(np.array_equal(a['a_j'],b['a_j']))
    same_base=a['base_trials']==b['base_trials']
    invariance.append({
        'detector':d,'same_base_trials':same_base,'same_A_B_membership':same_ab,
        'same_A_J_membership':same_aj,
        'coverage_budget_freeze':a['coverage_budget'],'coverage_budget_corrmix':b['coverage_budget'],
        'coverage_joint_freeze':a['coverage_joint'],'coverage_joint_corrmix':b['coverage_joint'],
    })
    if not (same_ab and same_aj and same_base):
        raise RuntimeError(f'Clean-workload invariance failed for {d}')
write_csv(OUT/'clean_workload_family_invariance.csv',invariance)

clean19=[]
for d in sorted(freeze_on):
    m=freeze_on[d]
    clean19.append({
        'detector':d,'panel':'attacked','canonical_source_family':'freeze',
        'n_total':m['n_total'],'n_A_B':m['n_A_B'],'n_A_J':m['n_A_J'],
        'coverage_budget':m['coverage_budget'],'coverage_joint':m['coverage_joint'],
    })
write_csv(MAN/'clean_workload_menu_19.csv',clean19)
positive12=[r for r in clean19 if r['coverage_budget']>0]
if len(positive12)!=12: raise RuntimeError(f'Expected 12 positive-budget detectors; got {len(positive12)}')
write_csv(MAN/'positive_budget_menu_12.csv',positive12)

coverage_summary=[
    {'menu':'full_fixed_19','n_configurations':19,
     'median_budget_coverage':median(r['coverage_budget'] for r in clean19),
     'median_joint_coverage':median(r['coverage_joint'] for r in clean19),
     'median_absolute_drop_budget_minus_joint':median(r['coverage_budget'] for r in clean19)-median(r['coverage_joint'] for r in clean19),
     'interpretation':'Median over all 19 detector configurations.'},
    {'menu':'fixed_observed_positive_budget_12','n_configurations':12,
     'median_budget_coverage':median(r['coverage_budget'] for r in positive12),
     'median_joint_coverage':median(r['coverage_joint'] for r in positive12),
     'median_absolute_drop_budget_minus_joint':median(r['coverage_budget'] for r in positive12)-median(r['coverage_joint'] for r in positive12),
     'interpretation':'Median over the 12 detector configurations with positive budget coverage in the canonical analysis.'},
]
write_csv(OUT/'coverage_aggregation_point_estimates.csv',coverage_summary)

# 38 incident-dependent configurations and 22 configurations defined under both recalibration settings.
incident38=[]
for kind in attack_kinds:
    for d in sorted(freeze_on):
        m=on[(kind,'score_only',d)]
        incident38.append({'kind':kind,'detector':d,'panel':'attacked','n_A_B_ON':m['n_A_B']})
write_csv(MAN/'incident_dependent_menu_38.csv',incident38)

intersection22=[]
for kind in attack_kinds:
    for d in sorted(freeze_on):
        mo=on[(kind,'score_only',d)]; mf=off[(kind,'score_only',d)]
        if mo['n_A_B']>0 and mf['n_A_B']>0:
            intersection22.append({'kind':kind,'detector':d,'panel':'attacked','n_A_B_ON':mo['n_A_B'],'n_A_B_OFF':mf['n_A_B']})
if len(intersection22)!=22: raise RuntimeError(f'Expected 22 rows defined under both settings, got {len(intersection22)}')
write_csv(MAN/'recalibration_shared_menu_22.csv',intersection22)

# Base-trial manifest from a canonical row.
ref=freeze_on[sorted(freeze_on)[0]]
base_rows=[]
for pos,(bid,row) in enumerate(zip(ref['base_trials'],ref['rows'])):
    base_rows.append({'position':pos,'base_trial':bid,'incident_start':row.get('incident_start'),'incident_end':row.get('incident_end')})
if len(base_rows)!=80: raise RuntimeError('Expected 80 canonical base trials')
write_csv(MAN/'base_trial_manifest_80.csv',base_rows)

# Base-trial pairing validation.
pairing=[]
reference_ids=ref['base_trials']
reference_windows={(r['base_trial'],r.get('incident_start'),r.get('incident_end')) for r in ref['rows']}
for condition,metrics in [('ON',on),('OFF',off)]:
    for kind in attack_kinds:
        for d in sorted(freeze_on):
            m=metrics[(kind,'score_only',d)]
            windows={(r['base_trial'],r.get('incident_start'),r.get('incident_end')) for r in m['rows']}
            pairing.append({
                'condition':condition,'kind':kind,'detector':d,'n_rows':m['n_total'],
                'n_unique_base_trials':len(set(m['base_trials'])),
                'base_trial_set_matches_reference':set(m['base_trials'])==set(reference_ids),
                'base_trial_order_matches_reference':m['base_trials']==reference_ids,
                'incident_windows_match_reference':windows==reference_windows,
            })
if not all(r['n_rows']==80 and r['n_unique_base_trials']==80 and r['base_trial_set_matches_reference'] and r['base_trial_order_matches_reference'] and r['incident_windows_match_reference'] for r in pairing):
    raise RuntimeError('Base-trial pairing validation failed')

# Four fixed main operating rows.
main_specs=[
    ('Fused Fisher','freeze','score_only','fused_fisher'),
    ('Coherence residual T2','freeze','score_only','coh_resid_t2'),
    ('CUSUM Factor LRT','freeze','score_only','cusum_factor_cov_lrt'),
    ('Max Abs control','freeze','all_only','max_abs'),
]
main_rows=[]
selected_manifest=[]
for label,kind,panel,det in main_specs:
    m=on[(kind,panel,det)]
    row=metric_public_row('cfpb_online_recal1_seed0',m)
    row={'display_row':label,**row}
    main_rows.append(row)
    selected_manifest.append({'display_row':label,'run':'cfpb_online_recal1_seed0','kind':kind,'panel':panel,'detector':det,'budget':m['spec'].budget_nominal,'strength':m['spec'].attack_strength})
write_csv(OUT/'main_operating_rows_point_estimates.csv',main_rows)
write_csv(MAN/'selected_main_rows_4.csv',selected_manifest)

main_table=[]
appendix_table=[]
for r in main_rows:
    main_table.append({k:r[k] for k in [
        'display_row','kind','panel','detector','n_total','n_A_B','n_A_J',
        'matched_burden_timely_at_k','matched_burden_contract_pass_rate','coverage_joint',
        'mean_tiw_clean_matched','mean_tiw_corrupted_matched','mean_cad_win_matched']})
    appendix_table.append({k:r[k] for k in [
        'display_row','kind','panel','detector','n_total','n_A_B','n_A_J','n_timely_A_B',
        'n_matched_contract_pass','n_all_trial_contract_pass','all_trial_contract_pass_rate',
        'mean_false_alert_events_corrupted_matched','mean_tiw_clean_matched','mean_tiw_corrupted_matched']})
write_csv(OUT/'main_operating_table_point_estimates.csv',main_table)
write_csv(OUT/'main_operating_appendix_point_estimates.csv',appendix_table)

# Recalibration point estimates. Difference of fixed-menu medians is primary.
def med_metric(metrics, keys, field):
    return median(metrics[k][field] for k in keys)

def paired_median_change(keys, field):
    return median(on[k][field]-off[k][field] for k in keys)

clean_keys=[('freeze','score_only',d) for d in sorted(freeze_on)]
incident_keys=[(kind,'score_only',d) for kind in attack_kinds for d in sorted(freeze_on)]
common_keys=[(r['kind'],'score_only',r['detector']) for r in intersection22]
recal=[]
for name,keys,field in [
    ('budget_coverage_median_19',clean_keys,'coverage_budget'),
    ('joint_coverage_median_19',clean_keys,'coverage_joint'),
    ('all_trial_contract_pass_median_38',incident_keys,'all_trial_contract_pass_rate'),
    ('matched_burden_timely_median_22',common_keys,'matched_burden_timely_at_k'),
    ('matched_burden_contract_pass_median_22',common_keys,'matched_burden_contract_pass_rate'),
]:
    onv=med_metric(on,keys,field); offv=med_metric(off,keys,field)
    recal.append({
        'estimand':name,'configuration_menu_size':len(keys),'on_value':onv,'off_value':offv,
        'difference_of_fixed_menu_medians_ON_minus_OFF':onv-offv,
        'median_paired_configuration_change_ON_minus_OFF':paired_median_change(keys,field),
        'primary_contrast':'difference_of_fixed_menu_medians_ON_minus_OFF',
    })

on_positive={d for d in sorted(freeze_on) if freeze_on[d]['coverage_joint']>0}
off_positive={d for d in sorted(freeze_off) if freeze_off[d]['coverage_joint']>0}
transitions={
    '0_to_0':len(set(freeze_on)-on_positive-off_positive),
    '0_to_1_OFF_to_ON':len(on_positive-off_positive),
    '1_to_0_OFF_to_ON':len(off_positive-on_positive),
    '1_to_1':len(on_positive&off_positive),
}
recal.append({
    'estimand':'nonzero_joint_fraction_19','configuration_menu_size':19,
    'on_value':len(on_positive)/19,'off_value':len(off_positive)/19,
    'difference_of_fixed_menu_medians_ON_minus_OFF':(len(on_positive)-len(off_positive))/19,
    'median_paired_configuration_change_ON_minus_OFF':float('nan'),
    'primary_contrast':'change_in_fraction; paired transition counts reported separately',
})
write_csv(OUT/'recalibration_point_estimates.csv',recal)
write_csv(OUT/'recalibration_nonzero_joint_transitions.csv',[{
    **transitions,'n_detectors':19,'on_nonzero_count':len(on_positive),'off_nonzero_count':len(off_positive),
    'on_fraction':len(on_positive)/19,'off_fraction':len(off_positive)/19,
}])

# Clean regime pass remains descriptive. Use the stored drift slice summary, which
# explicitly defines the fixed clean-timeline scope used by the manuscript (attacked
# slice, budgets B>=0.75; n_rows=285).
def clean_regime_descriptive(run_name):
    path=DATA_ROOT/run_name/'drift_slice_summary_paper.csv'
    rows=read_csv(path)
    candidates=[r for r in rows if r.get('slice')=='score_only_budget_ge_0p75']
    if not candidates:
        raise RuntimeError(f'Missing fixed-timeline drift slice for {run_name}')
    # clean_pass_rate_mean is invariant across drift deltas within a run.
    vals={float(r['clean_pass_rate_mean']) for r in candidates}
    nrows={int(float(r['n_rows'])) for r in candidates}
    if len(vals)!=1 or len(nrows)!=1:
        raise RuntimeError(f'Clean fixed-timeline summary is not invariant in {run_name}')
    return vals.pop(), nrows.pop(), str(path.relative_to(ROOT))
reg_on,n_on,src_on=clean_regime_descriptive('cfpb_online_recal1_seed0')
reg_off,n_off,src_off=clean_regime_descriptive('cfpb_online_recal0_seed0')
write_csv(OUT/'clean_regime_pass_descriptive.csv',[{
    'timeline':'fixed observed clean timeline','scope':'attacked slice; budgets B>=0.75',
    'recalibration_ON_mean_pass':reg_on,
    'recalibration_OFF_mean_pass':reg_off,'difference_ON_minus_OFF':reg_on-reg_off,
    'n_aligned_configuration_window_rows_ON':n_on,'n_aligned_configuration_window_rows_OFF':n_off,
    'source_ON':src_on,'source_OFF':src_off,
    'uncertainty_treatment':'descriptive only; no interval from incident-trial bootstrap',
}])

# Sensitivity table: clean quantities once over 19 detectors; incident-dependent metrics over 38 rows.
sensitivity_runs=[
    ('cfpb_online_recal1_seed0','Canonical recalibration ON'),
    ('cfpb_recal_win36_seed0','Window 36'),
    ('cfpb_recal_eta015_seed0','Gain 0.15'),
    ('cfpb_recal_upd3_seed0','Update every 3'),
    ('cfpb_recal_maxup110_seed0','Max up 1.10'),
]
sens=[]
for run,label in sensitivity_runs:
    mm=all_metrics[run]
    fm={m['spec'].detector:m for m in mm.values() if m['spec'].panel=='score_only' and m['spec'].kind=='freeze'}
    cm={m['spec'].detector:m for m in mm.values() if m['spec'].panel=='score_only' and m['spec'].kind=='corr_mix'}
    if set(fm)!=set(cm) or len(fm)!=19: raise RuntimeError(f'Bad clean menu in {run}')
    for d in fm:
        if not (np.array_equal(fm[d]['a_b'],cm[d]['a_b']) and np.array_equal(fm[d]['a_j'],cm[d]['a_j'])):
            raise RuntimeError(f'Clean family mismatch {run} {d}')
    incident=[m for m in mm.values() if m['spec'].panel=='score_only' and m['spec'].kind in attack_kinds]
    defined=[m for m in incident if m['n_A_B']>0]
    clean=list(fm.values())
    positive=[m for m in clean if m['coverage_budget']>0]
    sens.append({
        'run':run,'setting':label,
        'n_clean_detectors':len(clean),'n_clean_positive_budget':len(positive),
        'median_budget_coverage_full_19':median(m['coverage_budget'] for m in clean),
        'median_joint_coverage_full_19':median(m['coverage_joint'] for m in clean),
        'median_budget_coverage_positive_menu':median(m['coverage_budget'] for m in positive),
        'median_joint_coverage_positive_menu':median(m['coverage_joint'] for m in positive),
        'nonzero_joint_count_19':sum(m['coverage_joint']>0 for m in clean),
        'nonzero_joint_fraction_19':sum(m['coverage_joint']>0 for m in clean)/19,
        'n_incident_rows_total':len(incident),'n_incident_rows_defined_A_B':len(defined),
        'median_MB_timely_defined_rows':median(m['matched_burden_timely_at_k'] for m in defined),
        'median_MB_contract_pass_defined_rows':median(m['matched_burden_contract_pass_rate'] for m in defined),
        'median_all_trial_contract_pass_38':median(m['all_trial_contract_pass_rate'] for m in incident),
        'mean_corrupted_TIW_defined_rows':float(np.mean([m['mean_tiw_corrupted_matched'] for m in defined])),
    })
write_csv(OUT/'recalibration_sensitivity_point_estimates.csv',sens)

# TIW-cap sensitivity point estimates using the same 19/38 distinction.
tiw_sens=[]
for run,label in [('cfpb_online_recal1_tiw010_seed0','TIW cap 0.10'),('cfpb_online_recal1_seed0','TIW cap 0.15 canonical'),('cfpb_online_recal1_tiw020_seed0','TIW cap 0.20')]:
    mm=all_metrics[run]
    clean=[m for m in mm.values() if m['spec'].panel=='score_only' and m['spec'].kind=='freeze']
    incident=[m for m in mm.values() if m['spec'].panel=='score_only' and m['spec'].kind in attack_kinds]
    defined=[m for m in incident if m['n_A_B']>0]
    tiw_sens.append({
        'run':run,'setting':label,'tiw_cap':clean[0]['tiw_cap'],'n_clean_detectors':len(clean),
        'median_budget_coverage_full_19':median(m['coverage_budget'] for m in clean),
        'median_joint_coverage_full_19':median(m['coverage_joint'] for m in clean),
        'nonzero_joint_count_19':sum(m['coverage_joint']>0 for m in clean),
        'nonzero_joint_fraction_19':sum(m['coverage_joint']>0 for m in clean)/19,
        'n_incident_rows_defined_A_B':len(defined),
        'median_MB_contract_pass_defined_rows':median(m['matched_burden_contract_pass_rate'] for m in defined),
    })
write_csv(OUT/'tiw_cap_sensitivity_point_estimates.csv',tiw_sens)

# Cross-dataset selectors under the prespecified rule.
selector_runs=[
    'cfpb_online_recal1_seed0','smap_A1_weekly_ledger_seed0','smap_D15_weekly_ledger_seed0',
    'cfpb_gradual_bias_seed0','smap_A1_gradual_bias_seed0','smap_D15_gradual_bias_seed0',
]
selected=[]
ties=[]
for run in selector_runs:
    mm=all_metrics[run]
    kinds=sorted({k for k,p,d in mm if p=='score_only'})
    for kind in kinds:
        candidates=[m for (k,p,d),m in mm.items() if k==kind and p=='score_only' and m['n_A_B']>0]
        if not candidates: continue
        max_timely=max(m['matched_burden_timely_at_k'] for m in candidates)
        naive=[m for m in candidates if np.isclose(m['matched_burden_timely_at_k'],max_timely)]
        max_cp=max(m['matched_burden_contract_pass_rate'] for m in candidates)
        c1=[m for m in candidates if np.isclose(m['matched_burden_contract_pass_rate'],max_cp)]
        max_tie=max(m['matched_burden_timely_at_k'] for m in c1)
        compliant=[m for m in c1 if np.isclose(m['matched_burden_timely_at_k'],max_tie)]
        if len(naive)>1: ties.append({'run':run,'kind':kind,'selector':'naive','detectors':'|'.join(sorted(m['spec'].detector for m in naive))})
        if len(compliant)>1: ties.append({'run':run,'kind':kind,'selector':'best_matched_compliant','detectors':'|'.join(sorted(m['spec'].detector for m in compliant))})
        if len(naive)!=1 or len(compliant)!=1:
            raise RuntimeError(f'Unresolved selector tie in {run}/{kind}')
        n=naive[0]; c=compliant[0]
        selected.append({
            'run':run,'dataset':RUN_META[run]['dataset'],'kind':kind,
            'naive_detector':n['spec'].detector,'naive_MB_timely':n['matched_burden_timely_at_k'],
            'naive_MB_contract_pass':n['matched_burden_contract_pass_rate'],'naive_n_A_B':n['n_A_B'],
            'best_matched_compliant_detector':c['spec'].detector,
            'compliant_MB_timely':c['matched_burden_timely_at_k'],
            'compliant_MB_contract_pass':c['matched_burden_contract_pass_rate'],
            'compliant_all_trial_contract_pass':c['all_trial_contract_pass_rate'],
            'compliant_n_A_B':c['n_A_B'],'compliant_n_A_J':c['n_A_J'],
            'compliant_mean_clean_TIW':c['mean_tiw_clean_matched'],
            'compliant_mean_corrupted_TIW':c['mean_tiw_corrupted_matched'],
            'compliant_CAD':c['mean_cad_win_matched'],
        })
write_csv(OUT/'cross_dataset_selector_point_estimates.csv',selected)
write_csv(OUT/'selector_tie_diagnostics.csv',ties,fieldnames=['run','kind','selector','detectors'])

# Silent suppression exact reconstruction: canonical CFPB, attacked Freeze, max strength, fixed 12 positive-budget menu.
silent_trials=[]
counts={'no_alert_stream_change':0,'changed_with_warning_overlap':0,'silent_suppression':0}
for menu_row in positive12:
    d=menu_row['detector']; m=freeze_on[d]
    for r in m['rows']:
        cad=float(r['cad_win']); event_success=int(float(r['event_success']))
        if cad<=0:
            category='no_alert_stream_change'
        elif event_success==1:
            category='changed_with_warning_overlap'
        else:
            category='silent_suppression'
        counts[category]+=1
        silent_trials.append({
            'detector':d,'base_trial':r['base_trial'],'incident_start':r.get('incident_start'),
            'incident_end':r.get('incident_end'),'cad_win':cad,'event_success':event_success,'category':category,
        })
if len(silent_trials)!=960 or counts['silent_suppression']!=112:
    raise RuntimeError(f'Unexpected silent suppression reconstruction: n={len(silent_trials)}, counts={counts}')
write_csv(OUT/'silent_suppression_trial_classification.csv',silent_trials)
write_csv(OUT/'silent_suppression_summary.csv',[{
    'run':'cfpb_online_recal1_seed0','kind':'freeze','panel':'attacked','budget':1.0,'attack_strength':1.0,
    'detector_menu':'fixed observed positive-budget menu','n_detectors':12,'n_unique_base_trials':80,
    'n_pooled_detector_trials':960,
    'n_no_alert_stream_change':counts['no_alert_stream_change'],
    'fraction_no_alert_stream_change':counts['no_alert_stream_change']/960,
    'n_changed_with_warning_overlap':counts['changed_with_warning_overlap'],
    'fraction_changed_with_warning_overlap':counts['changed_with_warning_overlap']/960,
    'n_silent_suppression':counts['silent_suppression'],
    'fraction_silent_suppression':counts['silent_suppression']/960,
    'interpretation':'Pooled descriptive rate across detector configurations and base trials.',
}])
write_csv(MAN/'silent_suppression_detector_menu_12.csv',[{'detector':r['detector']} for r in positive12])

# Save run and output checksums.
(MAN/'source_run_manifests.json').write_text(json.dumps(run_manifests,indent=2,sort_keys=True),encoding='utf-8')
point_output_files = [
    p for p in sorted(OUT.glob('*.csv'))
    if not p.name.startswith(('core_cfpb_', 'cfpb_gradualbias_'))
    and p.name != 'selected_main_rows_pairing_assertion.csv'
]
files = point_output_files + [
    p for p in sorted(MAN.glob('*.csv'))
    if p.name != 'point_estimate_artifact_checksums.csv'
]
checksums=[{'relative_path':str(p.relative_to(ROOT)),'sha256':sha256_file(p),'bytes':p.stat().st_size} for p in files]
write_csv(MAN/'point_estimate_artifact_checksums.csv',checksums)

summary={
    'status':'PASS','n_point_rows':len(all_public),'n_clean_detectors':19,'n_positive_budget_detectors':12,
    'n_incident_rows':38,'n_rows_defined_under_both_settings':22,'n_base_trials':80,
    'coverage_full19':coverage_summary[0],'coverage_positive12':coverage_summary[1],
    'nonzero_joint_ON':'8/19','nonzero_joint_OFF':f'{len(off_positive)}/19',
    'silent_suppression':'112/960',
    'clean_regime_pass_descriptive':{'ON':reg_on,'OFF':reg_off},
}
(OUT/'point_estimate_build_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True),encoding='utf-8')
print(json.dumps(summary,indent=2,sort_keys=True))
