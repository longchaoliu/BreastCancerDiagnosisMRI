#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds a timing table for breast MR series by scanning multiple root directories
for DICOM series, extracting per-series metadata, and writing a CSV compatible
with geninputs.py expectations.

Output: /FL_system/data/Data_table_timing.csv

Columns (at minimum):
- SessionID: stable identifier per study (PatientID_StudyDate_StudyTime)
- SeriesInstanceUID
- StudyInstanceUID
- PatientID
- Major: per-session order index (ascending AcquisitionDateTime/SeriesTime)
- TriTime: Trigger time in ms if available, else 'Unknown'
- ScanDur: Acquisition duration in us if available, else 'Unknown'
- AcqTime: acquisition time as HH:MM:SS
- Series_desc: SeriesDescription
- Modality
- BodyPart
- DicomPath: directory containing this series (first detected instance)

This utility is conservative and resilient to mixed directory structures. It
parses only one instance per series to minimize IO, and assigns ordering after
gathering all series in a session.
"""

import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


try:
    import pydicom as pyd
    from pydicom.errors import InvalidDicomError
except Exception as exc:  # pragma: no cover
    print('ERROR: pydicom is required to run this script.\n', exc)
    sys.exit(1)


ROOTS = [
    '/mnt/shareddata/datasets/AIR/extracted/',
    '/mnt/shareddata/datasets/YAIB-cohorts/AIR_API_Downloads/missing_breast_mri_feb9_24/',
    '/mnt/shareddata/datasets/YAIB-cohorts/AIR_API_Downloads/chest_ct_breast_mr_missing',
]

OUTPUT_DIR = 'data'
OUTPUT_PATH = f'{OUTPUT_DIR}/Data_table_timing.csv'


def is_probably_dicom(file_path: str) -> bool:
    """Lightweight DICOM detection to avoid expensive reads."""
    try:
        
        with open(file_path, 'rb') as f:
            prefix = f.read(132)
            return len(prefix) >= 132 and prefix[128:132] == b'DICM'
    except Exception:
        return False


def try_read_header(file_path: str):
    """Read a DICOM header without pixel data. Returns dataset or None."""
    try:
        return pyd.dcmread(file_path, stop_before_pixels=True, force=True)
    except (InvalidDicomError, Exception):
        return None


def hhmmss_to_hh_mm_ss(hhmmss: str) -> Optional[str]:
    """Convert DICOM time 'HHMMSS(.ffffff)' to 'HH:MM:SS'."""
    if not hhmmss:
        return None
    try:
        base = hhmmss.split('.')[0]
        base = base.zfill(6)
        return f"{base[0:2]}:{base[2:4]}:{base[4:6]}"
    except Exception:
        return None


def extract_series_record(file_path: str, ds) -> Optional[Dict]:
    """Extract per-series metadata from a single instance."""
    try:
        modality = getattr(ds, 'Modality', None)
        if modality and str(modality).upper() != 'MR':
            return None

        series_uid = getattr(ds, 'SeriesInstanceUID', None)
        study_uid = getattr(ds, 'StudyInstanceUID', None)
        patient_id = getattr(ds, 'PatientID', None)
        series_desc = getattr(ds, 'SeriesDescription', None)
        body_part = getattr(ds, 'BodyPartExamined', None)
        series_num = getattr(ds, 'SeriesNumber', None)
        accession = getattr(ds, 'AccessionNumber', None)

        acq_date = getattr(ds, 'AcquisitionDate', None) or getattr(ds, 'StudyDate', None)
        acq_time = getattr(ds, 'AcquisitionTime', None) or getattr(ds, 'SeriesTime', None) or getattr(ds, 'ContentTime', None)
        acq_time_hms = hhmmss_to_hh_mm_ss(str(acq_time)) if acq_time else None

        tri_time = getattr(ds, 'TriggerTime', None)
        if tri_time is None:
            tri_time_val = 'Unknown'
        else:
            try:
                tri_time_val = float(tri_time)
            except Exception:
                tri_time_val = 'Unknown'

        acq_duration = getattr(ds, 'AcquisitionDuration', None)
        if acq_duration is None:
            scan_dur = 'Unknown'
        else:
            try:
                # Convert seconds -> microseconds to match expectations in geninputs.py path
                scan_dur = int(float(acq_duration) * 1e6)
            except Exception:
                scan_dur = 'Unknown'

        # SessionID based on PatientID + StudyDate + StudyTime if available, else StudyInstanceUID
        study_time = getattr(ds, 'StudyTime', None)
        study_time_hms = hhmmss_to_hh_mm_ss(str(study_time)) if study_time else None
        if patient_id and acq_date and study_time_hms:
            session_id = f"{str(patient_id)}_{str(acq_date)}_{study_time_hms.replace(':','')}"
        elif study_uid:
            session_id = str(study_uid)
        else:
            # Fallback: bucket by top directory name
            session_id = os.path.basename(os.path.dirname(file_path))

        record = {
            'SessionID': session_id,
            'SeriesInstanceUID': str(series_uid) if series_uid is not None else None,
            'StudyInstanceUID': str(study_uid) if study_uid is not None else None,
            'PatientID': str(patient_id) if patient_id is not None else None,
            'SeriesNumber': int(series_num) if isinstance(series_num, (int, float)) else None,
            'AccessionNumber': str(accession) if accession is not None else None,
            'TriTime': tri_time_val,
            'ScanDur': scan_dur,
            'AcqDate': str(acq_date) if acq_date is not None else None,
            'AcqTime': acq_time_hms if acq_time_hms is not None else 'Unknown',
            'Modality': str(modality) if modality is not None else None,
            'DicomPath': file_path,
        }
        return record
    except Exception:
        return None


def scan_series(roots: List[str], path_substr: Optional[str] = None) -> List[Dict]:
    """Walk roots, pick one instance per SeriesInstanceUID, and extract metadata.
    Optionally restrict to directories containing path_substr.
    """
    seen_series: set = set()
    records: List[Dict] = []
    for root in roots:
        if not os.path.exists(root):
            continue
        filenames = os.listdir(f"{root}/{path_substr}")

        # Heuristic: prioritize files that look like DICOM
        candidates = [f for f in filenames if f.lower().endswith('.dcm') or f.upper().startswith('IM') or f.upper().startswith('MR')]
        if not candidates:
            candidates = filenames
        for fname in tqdm(candidates[:100]):
            fpath = os.path.join(f"{root}/{path_substr}", fname)
            ds = try_read_header(fpath)
            if ds is None:
                continue
            series_uid = getattr(ds, 'SeriesInstanceUID', None)
            if not series_uid:
                continue
            if series_uid in seen_series:
                continue
            rec = extract_series_record(fpath, ds)
            if rec is None:
                continue
            records.append(rec)
            seen_series.add(series_uid)
    return records


def assign_major(records: List[Dict]) -> List[Dict]:
    """Assign per-session order based on AcquisitionDate+Time, then SeriesNumber."""
    by_session: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        if r.get('SessionID') is None:
            continue
        by_session[str(r['SessionID'])].append(r)

    for session_id, rows in by_session.items():
        def key_fn(r):
            date = r.get('AcqDate') or ''
            time = r.get('AcqTime') or '00:00:00'
            try:
                dt = datetime.strptime(f"{date} {time}", "%Y%m%d %H:%M:%S")
            except Exception:
                dt = datetime.min
            series_num = r.get('SeriesNumber') or 0
            return (dt, series_num)

        rows.sort(key=key_fn)
        for idx, r in enumerate(rows):
            r['Major'] = int(idx)

    # Flatten
    out = []
    for rows in by_session.values():
        out.extend(rows)
    return out


def write_csv(records: List[Dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame.from_records(records)
    # Reorder/select columns to match expectations and readability
    cols = [
        'SessionID', 'Major', 'PatientID', 'StudyInstanceUID', 'SeriesInstanceUID',
        'SeriesNumber', 'AccessionNumber', 'AcqDate', 'AcqTime', 'TriTime', 'ScanDur',
        'Series_desc', 'Modality', 'BodyPart', 'DicomPath',
    ]
    df = df.reindex(columns=cols)
    
    metadata = pd.read_csv('/mnt/shareddata/datasets/breast_ucsf_mri/contrast_pixel_space/metadata/mri_series.csv')
    metadata = metadata[metadata['Orig Study UID'] == df.iloc[0]['StudyInstanceUID']]
    df = df.merge(metadata, left_on='SeriesNumber', right_on='Orig Series #' )
    df.to_csv(output_path, index=False)
    print(f'Wrote timing table with {len(df)} rows to: {output_path}')


def filter_records(records: List[Dict], accession: Optional[str], patient: Optional[str], session: Optional[str], path_substr: Optional[str]) -> List[Dict]:
    def keep(r: Dict) -> bool:
        if accession and str(r.get('AccessionNumber', '')) != str(accession):
            return False
        if patient and str(r.get('PatientID', '')) != str(patient):
            return False
        if session and str(r.get('SessionID', '')) != str(session):
            return False
        if path_substr and path_substr not in str(r.get('DicomPath', '')):
            return False
        return True
    return [r for r in records if keep(r)]


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Index DICOM MR series and build timing CSV')
    parser.add_argument('--roots', nargs='*', default=ROOTS, help='Root directories to scan')
    parser.add_argument('--output', default=OUTPUT_PATH, help='Output CSV path')
    parser.add_argument('--filter-accession', default=None, help='AccessionNumber to include')
    parser.add_argument('--filter-patient', default=None, help='PatientID to include')
    parser.add_argument('--filter-session', default=None, help='SessionID to include')
    parser.add_argument('--filter-path-substr', default=None, help='Only include series whose path contains this substring')
    args = parser.parse_args()

    roots = args.roots if args.roots else ROOTS
    records = scan_series(roots, path_substr=args.filter_path_substr)
    if not records:
        print('No MR series found in provided roots.')
        return
    # records = filter_records(records, args.filter_accession, args.filter_patient, args.filter_session, args.filter_path_substr)
    # if not records:
    #     print('No series matched filters.')
    #     return
    records = assign_major(records)
    write_csv(records, args.output)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted by user')
    except Exception as exc:
        print('Failed to build timing table:', exc)
        traceback.print_exc()

