"""Freeze an explicitly selected validation configuration before official test."""
import argparse,json
from pathlib import Path
from .prepare import filehash

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--validation-report',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 report=json.loads(Path(a.validation_report).read_text());manifest=json.loads(Path(a.manifest).read_text())
 if report.get('split')!='validation' or report.get('manifest_sha256')!=filehash(a.manifest) or report.get('dataset')!=manifest['dataset'] or report.get('failures')!=0 or report.get('query_count_recorded')!=manifest['validation']['count'] or len(report.get('results',[]))!=manifest['validation']['count'] or {x.get('query_id') for x in report.get('results',[])}!=set(manifest['validation']['query_ids']) or any(x.get('failure') is not None for x in report.get('results',[])) or report.get('eligibility')!='comparable':raise ValueError('Selection requires a complete, zero-failure validation report')
 out={'version':1,'pipeline_config_sha256':report.get('pipeline_config_sha256'),'manifest_sha256':filehash(a.manifest),'config':report['config'],'code_sha256':report['code_sha256'],'validation_report_sha256':filehash(a.validation_report),'validation_failures':0,'selection_reason':'Explicit operator selection from frozen validation; official test not used to select configuration.'}
 target=Path(a.output)
 if target.exists():raise FileExistsError('Selection exists; preserve prior decision')
 target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'selection_sha256':filehash(target)}))
if __name__=='__main__':main()
