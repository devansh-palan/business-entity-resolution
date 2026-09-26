import os
import polars as pl, json, gc, time, sys
SCR = os.environ.get("BER_SCRATCH", "artifacts/_scratch")   # parquet cache of the raw TSVs, built by s0_inventory.py
FILES = ["train_source1","train_source2","train_source3","test_source1","test_source2","test_source3"]
LEGAL = set("inc incorporated llc corp corporation co company ltd limited pvt private llp plc lp pc pllc sarl sas sa eurl sci snc scp sasu gmbh".split())
Q = [0.05,0.25,0.5,0.75,0.95,0.99]

def norm(c):
    e = pl.col(c).fill_null("").str.to_lowercase()
    try:
        e = e.str.normalize("NFKD").str.replace_all(r"\p{M}+", "")
    except Exception:
        pass
    return e.str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()

def qstats(s):
    s = s.drop_nulls()
    return {"mean": round(float(s.mean()),2), **{f"p{int(q*100)}": float(s.quantile(q)) for q in Q}, "max": int(s.max())}

def topn(df, col, n, by="country"):
    out = {}
    for ctry, sub in df.group_by(by):
        t = sub.group_by(col).len().sort("len", descending=True).head(n)
        out[ctry[0]] = [(v, int(c)) for v, c in zip(t[col], t["len"])]
    return out

OUT = {}
for key in FILES:
    t0 = time.time()
    df = pl.read_parquet(f"{SCR}/pq/{key}.parquet")
    N = df.height
    df = df.with_columns([
        pl.col("business_name").str.len_chars().alias("name_len"),
        pl.col("business_name").str.count_matches(r"\S+").alias("name_ntok"),
        pl.col("business_address").str.len_chars().alias("addr_len"),
        pl.col("business_address").str.count_matches(r"\S+").alias("addr_ntok"),
        (pl.col("business_address").str.count_matches(",")+1).alias("addr_ncomp"),
        pl.col("business_address").str.extract(r",\s*([^,]+)$",1).str.strip_chars().alias("addr_last"),
        pl.col("business_address").str.extract(r"^([^,]+),",1).str.strip_chars().alias("addr_first"),
        norm("business_name").alias("name_norm"),
        norm("business_address").alias("addr_norm"),
    ])
    LEG = list(LEGAL)
    df = df.with_columns([
        pl.col("name_norm").str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(LEG) & (pl.element()!=""))).list.join(" ").alias("name_core"),
        pl.col("name_norm").str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(LEG) & (pl.element()!=""))).list.sort().list.join(" ").alias("name_core_sorted"),
        pl.col("addr_norm").str.split(" ").list.eval(pl.element().filter(pl.element()!="")).list.sort().list.join(" ").alias("addr_sorted"),
        pl.col("name_norm").str.split(" ").list.last().alias("name_last_tok"),
        pl.col("name_norm").str.split(" ").list.first().alias("name_first_tok"),
    ])
    nm, ad = pl.col("business_name"), pl.col("business_address")
    flags = {
      "name_devanagari": nm.str.contains(r"[\u0900-\u097F]"),
      "name_nonascii": nm.str.contains(r"[^\x00-\x7F]"),
      "name_latin_accent": nm.str.contains(r"[\u00C0-\u024F]"),
      "name_allcaps": nm.str.contains(r"\p{Lu}") & ~nm.str.contains(r"\p{Ll}"),
      "name_alllower": nm.str.contains(r"\p{Ll}") & ~nm.str.contains(r"\p{Lu}"),
      "name_leading_punct": nm.str.contains(r"^[^\p{L}\p{N}]"),
      "name_trailing_punct": nm.str.contains(r"[^\p{L}\p{N}.)]$"),
      "name_has_digit": nm.str.contains(r"\d"),
      "name_has_amp": nm.str.contains(r"&"),
      "name_has_and": nm.str.contains(r"(?i)\band\b"),
      "name_has_paren": nm.str.contains(r"[()]"),
      "name_double_space": nm.str.contains(r"\s{2,}"),
      "name_lead_trail_space": nm.str.contains(r"^\s|\s$"),
      "name_dba": nm.str.contains(r"(?i)\b(dba|d/b/a|t/a|trading as|aka)\b"),
      "name_url": nm.str.contains(r"(?i)\.(com|net|org|in|fr|co|io|biz)\b"),
      "name_has_legal_suffix_token": pl.col("name_norm").str.split(" ").list.eval(pl.element().is_in(LEG)).list.any(),
      "addr_missing": ad.is_null(),
      "addr_6digit": ad.str.contains(r"\b\d{6}\b"),
      "addr_5digit": ad.str.contains(r"\b\d{5}\b"),
      "addr_zip4": ad.str.contains(r"\b\d{5}-\d{4}\b"),
      "addr_10digit": ad.str.contains(r"\b\d{10}\b"),
      "addr_any_digit": ad.str.contains(r"\d"),
      "addr_landmark": ad.str.contains(r"(?i)\b(near|nr|opp|opposite|behind|beside|next to|adjacent|in front of)\b"),
      "addr_pobox": ad.str.contains(r"(?i)\bp\.?\s?o\.?\s?box\b"),
      "addr_unit": ad.str.contains(r"(?i)\b(unit|apt|apartment|suite|ste|floor|flr|fl|bldg|building|room|rm|shop|office|no|h\.?no)\b|#"),
      "addr_devanagari": ad.str.contains(r"[\u0900-\u097F]"),
      "addr_nonascii": ad.str.contains(r"[^\x00-\x7F]"),
      "addr_latin_accent": ad.str.contains(r"[\u00C0-\u024F]"),
      "addr_allcaps": ad.str.contains(r"\p{Lu}") & ~ad.str.contains(r"\p{Ll}"),
      "addr_alllower": ad.str.contains(r"\p{Ll}") & ~ad.str.contains(r"\p{Lu}"),
      "addr_starts_digit": ad.str.contains(r"^\d"),
      "addr_us_abbrev": ad.str.contains(r"(?i)\b(rd|st|ave|blvd|dr|ln|ct|hwy|pkwy|cir|pl|trl|ter)\b"),
      "addr_us_full": ad.str.contains(r"(?i)\b(road|street|avenue|boulevard|drive|lane|court|highway|parkway|circle|place|trail|terrace)\b"),
      "addr_fr_street": ad.str.contains(r"(?i)\b(rue|r\.|av\.|bd|chemin|allee|allée|impasse|route|quai|cours|voie|faubourg)\b"),
      "addr_in_tokens": ad.str.contains(r"(?i)\b(nagar|marg|colony|sector|phase|block|plot|gali|chowk|bazar|bazaar|mandi|tehsil|taluk|tq|dist|distt|district|village|vill|mohalla|ward|kh no|khasra|survey)\b"),
      "addr_last_is_2cap": pl.col("addr_last").str.contains(r"^[A-Z]{2}$"),
      "addr_last_is_digits": pl.col("addr_last").str.contains(r"^\d+$"),
      "addr_double_space": ad.str.contains(r"\s{2,}"),
    }
    df = df.with_columns([v.alias(k) for k,v in flags.items()])
    res = {"n_rows": N}
    per = {}
    for (ctry,), sub in df.group_by("country"):
        d = {"n": sub.height}
        d["name_len"] = qstats(sub["name_len"]); d["name_ntok"] = qstats(sub["name_ntok"])
        d["addr_len"] = qstats(sub["addr_len"]); d["addr_ntok"] = qstats(sub["addr_ntok"]); d["addr_ncomp"] = qstats(sub["addr_ncomp"])
        d["flags_pct"] = {k: round(100*float(sub[k].fill_null(False).mean()),3) for k in flags}
        nc = sub.group_by("addr_ncomp").len().sort("addr_ncomp")
        d["addr_ncomp_dist"] = {str(k): int(v) for k,v in zip(nc["addr_ncomp"], nc["len"])}
        per[ctry] = d
    res["per_country"] = per
    res["top_last_name_tok"] = topn(df, "name_last_tok", 40)
    res["top_first_name_tok"] = topn(df, "name_first_tok", 30)
    toks = df.select("country", pl.col("name_norm").str.split(" ").alias("t")).explode("t").filter(pl.col("t")!="")
    res["top_name_tokens"] = topn(toks, "t", 60); del toks
    atoks = df.select("country", pl.col("addr_norm").str.split(" ").alias("t")).explode("t").filter(pl.col("t")!="")
    res["top_addr_tokens"] = topn(atoks, "t", 80); del atoks
    res["top_addr_last_comp"] = topn(df, "addr_last", 30)
    res["top_addr_first_comp"] = topn(df, "addr_first", 15)
    res["top_names_raw"] = topn(df, "business_name", 15)
    res["top_addr_raw"] = topn(df.filter(ad.is_not_null()), "business_address", 10)
    dup = {}
    dup["exact_row_dupes(name,addr,country)"] = N - df.n_unique(subset=["business_name","business_address","country"])
    for col in ["business_name","name_norm","name_core","name_core_sorted"]:
        g = df.group_by(col, "country").len()
        dup[f"{col}: rows sharing value w/ another row (same country)"] = int(g.filter(pl.col("len")>=2)["len"].sum())
        dup[f"{col}: distinct values"] = g.height
        dup[f"{col}: max group size"] = int(g["len"].max())
    ga = df.filter(ad.is_not_null()).group_by("addr_norm","country").len()
    dup["addr_norm: rows sharing value w/ another row"] = int(ga.filter(pl.col("len")>=2)["len"].sum()); dup["addr_norm: distinct values"] = ga.height; dup["addr_norm: max group size"] = int(ga["len"].max())
    gs = df.filter(ad.is_not_null()).group_by("addr_sorted","country").len()
    dup["addr_sorted: rows sharing value w/ another row"] = int(gs.filter(pl.col("len")>=2)["len"].sum())
    gna = df.group_by("name_core_sorted","addr_sorted","country").len()
    dup["(name_core_sorted, addr_sorted): rows sharing"] = int(gna.filter(pl.col("len")>=2)["len"].sum()); dup["(name_core_sorted, addr_sorted): max group size"] = int(gna["len"].max())
    gnl = df.group_by("name_core_sorted","addr_last","country").len()
    dup["(name_core_sorted, addr_last_component): rows sharing"] = int(gnl.filter(pl.col("len")>=2)["len"].sum())
    res["duplicates"] = {k: ((v, round(100*v/N,3)) if ("distinct" not in k and "max" not in k) else v) for k,v in dup.items()}
    gsz = df.group_by("name_core_sorted","country").len().with_columns(pl.col("len").clip(1,10).alias("k")).group_by("k").agg(pl.col("len").sum().alias("rows")).sort("k")
    res["name_core_sorted_groupsize_rows"] = {str(int(k)): int(v) for k,v in zip(gsz["k"], gsz["rows"])}
    df.select("entity_id","country","business_name","business_address","name_norm","name_core","name_core_sorted","addr_norm","addr_sorted","addr_last","addr_first","name_first_tok","name_devanagari","addr_missing","name_len","addr_len","name_ntok","addr_ntok").write_parquet(f"{SCR}/pq/{key}_norm.parquet")
    OUT[key] = res
    print(f"[{key}] done in {time.time()-t0:.0f}s", flush=True)
    del df; gc.collect()
json.dump(OUT, open(f"{SCR}/out/s3_text_profile.json","w",encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
print("DONE")
