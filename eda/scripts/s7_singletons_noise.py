import os
import polars as pl, json, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
SCR = os.environ.get("BER_SCRATCH", "artifacts/_scratch")   # parquet cache of the raw TSVs, built by s0_inventory.py
R = {}
cnt = pl.read_parquet(f"{SCR}/pq/train_gt_counts.parquet")
s1 = pl.read_parquet(f"{SCR}/pq/train_source1_norm.parquet", columns=["entity_id","country","name_core_sorted","addr_norm"]).join(cnt, left_on="entity_id", right_on="source1_entity_id")
lite = pl.concat([pl.read_parquet(f"{SCR}/pq/train_source{k}_norm.parquet", columns=["entity_id","country","name_core_sorted"]) for k in (2,3)])
blk = lite.group_by("country","name_core_sorted").len().rename({"len":"same_name_cands"})
s1 = s1.join(blk, on=["country","name_core_sorted"], how="left").with_columns(pl.col("same_name_cands").fill_null(0), (pl.col("n_total")==0).alias("singleton"))
g = s1.group_by("singleton").agg(pl.len().alias("n"), pl.col("same_name_cands").mean().round(1).alias("mean_same_name_cands"), pl.col("same_name_cands").median().alias("median"),
        (pl.col("same_name_cands")==0).mean().round(4).alias("pct_zero_same_name_cands"), (pl.col("same_name_cands")>=10).mean().round(4).alias("pct_ge10")).sort("singleton")
R["same_name_candidates_by_singleton_status"] = g.to_dicts()
# non-singletons: same-name candidates minus true matches = distractors that share the exact core name
ns = s1.filter(~pl.col("singleton")).with_columns((pl.col("same_name_cands") - pl.col("n_total")).clip(0, None).alias("distractors"))
R["nonsingleton_same_name_distractors_mean"] = round(float(ns["distractors"].mean()),1)
# ---- noise taxonomy rates on S2/S3 (train) by country, using raw text
for k in (2,3):
    d = pl.read_parquet(f"{SCR}/pq/train_source{k}.parquet")
    nm, ad = pl.col("business_name"), pl.col("business_address").fill_null("")
    ex = {
     "name_honorific_prefix(mr/smt/dr/sri/shri/m/s)": nm.str.contains(r"(?i)^(mr|mrs|smt|dr|sri|shri|m/s|ms)\.?\s"),
     "name_alias_marker(fka/formerly/dba/t/a/aka)": nm.str.contains(r"(?i)\b(fka|formerly|dba|d/b/a|t/a|aka|trading as)\b"),
     "name_domain_style(.com etc or no-space concat)": nm.str.contains(r"(?i)\.(com|net|org|in|fr|co|io|biz|c0m)\b|^#\S+$"),
     "name_digit_for_letter_typo(0 for o,1 for l)": nm.str.contains(r"[A-Za-z][01][A-Za-z]"),
     "name_leading_noise_punct(--,<<,***,##)": nm.str.contains(r"^[-<*#=~]+\s"),
     "name_two_letter_acronym_only": nm.str.contains(r"^[A-Z]{2,3}$"),
     "name_single_token": ~nm.str.strip_chars().str.contains(r"\s"),
     "name_extra_generic_token(service/center/partners appended)": nm.str.contains(r"(?i)\b(service|center|partners|group|holdings)$"),
     "addr_null_token(NULL/<NULL>/None/N A)": ad.str.contains(r"(?i)(<null>|\bnull\b|\bnone\b|\bn/?a\b)"),
     "addr_leading_zero_number": ad.str.contains(r"(^|[\s,#])0\d+"),
     "addr_hash_prefix(#)": ad.str.contains(r"#"),
     "addr_pobox_or_pmb_inserted": ad.str.contains(r"(?i)\b(po box|p\.o\. box|pmb)\b"),
     "addr_incorporated_or_llc_token_inserted": ad.str.contains(r"(?i)\b(incorporated|llc|inc)\b"),
     "addr_state_first_component": ad.str.contains(r"^[A-Z]{2},"),
     "addr_local_script_state": ad.str.contains(r"[\u0900-\u0DFF]"),
     "addr_door_no_prefix": ad.str.contains(r"(?i)^(door no|h\.?\s?no|house no)\b"),
     "addr_city_suffix_noise(CITY/CDP/Township)": ad.str.contains(r"(?i)\b(cdp|township|city)\b"),
     "addr_number_with_trailing_dot(17814.)": ad.str.contains(r"\d+\.\s"),
    }
    out = d.group_by("country").agg([v.mean().round(4).alias(n) for n, v in ex.items()]).sort("country")
    R[f"noise_rates_train_source{k}"] = out.to_dicts()
# ---- number perturbation among positives (sample feats): shared number vs not, by country and by address-missing
pos = pl.read_parquet(f"{SCR}/pq/sample_pos_feats.parquet")
R["positives_addr_num_overlap_by_country"] = pos.group_by("country_1").agg(pl.col("addr_num_overlap").mean().round(4), pl.col("addr_missing_2").mean().round(4), pl.len()).to_dicts()
R["positives_no_num_overlap_and_addr_present_pct"] = round(100*float((~pos["addr_num_overlap"] & ~pos["addr_missing_2"]).mean()),2)
# ---- how many S1 entities share their exact normalized address with another S1 entity (ambiguity if name is garbled)
sh = s1.group_by("country","addr_norm").len().filter(pl.col("len")>=2)
R["S1_entities_sharing_exact_addr_norm_with_other_S1"] = int(sh["len"].sum())
json.dump(R, open(f"{SCR}/out/s7.json","w",encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
for k, v in R.items():
    print("\n##", k)
    if isinstance(v, list):
        for r in v: print("   ", r)
    else: print("   ", v)
