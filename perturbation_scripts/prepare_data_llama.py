#!/usr/bin/env python3
"""
Data Preparation Script for Seed Entity Agent (Llama 3.1 Version)
Downloads and prepares all necessary data for offline entity linking
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from seed_entity_agent_llama import EntityDatabase


def prepare_freebase_wikidata_mapping():
    """Download and load Freebase/Wikidata mapping from HuggingFace."""
    print("\n" + "="*70)
    print("Step 1: Loading Freebase/Wikidata Mappings from HuggingFace")
    print("="*70)
    
    try:
        # Load from HuggingFace (auto-downloads)
        print("\nDownloading from HuggingFace...")
        print("URL: https://huggingface.co/datasets/kdm-daiict/freebase-wikidata-mapping")
        
        dataset = load_dataset("kdm-daiict/freebase-wikidata-mapping")
        
        print(f"✓ Loaded dataset")
        print(f"  Train split: {len(dataset['train'])} mappings")
        
        return dataset['train']
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Login to HuggingFace: huggingface-cli login")
        print("3. Try again")
        return None


def build_database(mapping_data):
    """Build SQLite database from mapping data."""
    print("\n" + "="*70)
    print("Step 2: Building SQLite Database")
    print("="*70)
    
    db_path = "data/entity_mappings.db"
    
    # Create database
    db = EntityDatabase(db_path)
    
    # Load mappings
    db.load_freebase_wikidata_mapping(mapping_data)
    
    db.close()
    
    print(f"\n✓ Database created: {db_path}")
    
    # Show file size
    db_file = Path(db_path)
    if db_file.exists():
        size_mb = db_file.stat().st_size / (1024 * 1024)
        print(f"  Size: {size_mb:.1f} MB")
    
    return True


def verify_datasets():
    """Verify that datasets are downloaded."""
    print("\n" + "="*70)
    print("Step 3: Verifying Datasets")
    print("="*70)
    
    datasets_dir = Path("data/datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    # Check WebQSP
    webqsp_file = datasets_dir / "WebQSP.test.json"
    if webqsp_file.exists():
        size_mb = webqsp_file.stat().st_size / (1024 * 1024)
        print(f"\n✓ WebQSP test set found ({size_mb:.1f} MB)")
        
        # Verify format
        try:
            with open(webqsp_file, 'r') as f:
                data = json.load(f)
                num_questions = len(data.get('Questions', []))
                print(f"  Questions: {num_questions}")
        except Exception as e:
            print(f"  ⚠ Error reading file: {e}")
    else:
        print(f"\n⚠ WebQSP test set not found")
        print(f"  Expected: {webqsp_file}")
        print(f"  Download from: https://www.microsoft.com/en-us/download/details.aspx?id=52763")
    
    # Check CWQ
    cwq_file = datasets_dir / "ComplexWebQuestions_test.json"
    if cwq_file.exists():
        size_mb = cwq_file.stat().st_size / (1024 * 1024)
        print(f"\n✓ CWQ test set found ({size_mb:.1f} MB)")
        
        # Verify format
        try:
            with open(cwq_file, 'r') as f:
                data = json.load(f)
                num_questions = len(data) if isinstance(data, list) else 1
                print(f"  Questions: {num_questions}")
        except Exception as e:
            print(f"  ⚠ Error reading file: {e}")
    else:
        print(f"\n⚠ CWQ test set not found")
        print(f"  Expected: {cwq_file}")
        print(f"\n  Download with:")
        print(f'  wget "https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AABr4ysSy_Tg8Wfxww4i_UWda/ComplexWebQuestions_test.json?dl=1" \\')
        print(f'    -O {cwq_file}')


def main():
    """Main setup flow."""
    print("\n" + "="*70)
    print("SEED ENTITY AGENT - DATA PREPARATION (Llama 3.1)")
    print("="*70)
    
    print("\nThis script will:")
    print("1. Download Freebase/Wikidata mappings from HuggingFace (~100 MB)")
    print("2. Build local SQLite database")
    print("3. Verify datasets are present")
    
    print("\nEstimated time: 5-10 minutes")
    print("Estimated disk space: ~500 MB")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Setup cancelled")
        return
    
    # Step 1: Load Freebase/Wikidata mapping
    mapping_data = prepare_freebase_wikidata_mapping()
    if not mapping_data:
        print("\n✗ Failed to load mapping data")
        return
    
    # Step 2: Build database
    success = build_database(mapping_data)
    if not success:
        print("\n✗ Failed to build database")
        return
    
    # Step 3: Verify datasets
    verify_datasets()
    
    # Summary
    print("\n" + "="*70)
    print("SETUP SUMMARY")
    print("="*70)
    
    print("\n✓ Freebase/Wikidata mapping loaded")
    print("✓ Database built")
    
    db_file = Path("data/entity_mappings.db")
    webqsp_file = Path("data/datasets/WebQSP.test.json")
    cwq_file = Path("data/datasets/ComplexWebQuestions_test.json")
    
    if db_file.exists() and webqsp_file.exists() and cwq_file.exists():
        print("\n✅ READY TO RUN!")
        print("\nNext steps:")
        print("  python seed_entity_agent_llama.py  # Test single query")
        print("  python run_evaluation_llama.py --dataset webqsp  # Full eval")
    else:
        print("\n⚠ SETUP INCOMPLETE")
        if not webqsp_file.exists():
            print("\n  Missing: WebQSP.test.json")
            print("  Download from: https://www.microsoft.com/en-us/download/details.aspx?id=52763")
        if not cwq_file.exists():
            print("\n  Missing: ComplexWebQuestions_test.json")
            print('  Run: wget "https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AABr4ysSy_Tg8Wfxww4i_UWda/ComplexWebQuestions_test.json?dl=1" -O data/datasets/ComplexWebQuestions_test.json')
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
