"""Persist assistant annotations from actual selected A1 texts, not human approval."""
from __future__ import annotations
import csv
import hashlib
import json
import logging
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/"src"))
from quality.reporting import read_table
from quality.vector_diagnostics import write_rows


def main():
    """Require exact selected IDs; save actual scope, comments and text hashes."""
    csv.field_size_limit(100000000)
    folder=PROJECT_ROOT/"outputs"/"quality"
    annotations=json.loads((folder/"A1_text_review_annotations.json").read_text(encoding="utf-8"))
    comments=annotations["annotations"]
    rows=[]
    for label in ["top20","bottom20"]:
        samples=read_table(folder/f"A1_{label}_quality_samples.csv")
        for rank,r in enumerate(samples,1):
            identity=r["id"]
            if identity not in comments:raise ValueError(f"New/unreviewed extreme text: {identity}")
            observations,judgement=comments[identity]
            # All top texts were read via complementary segments covering all
            # characters; short bottom texts were read entirely. Six long bottom
            # records were examined through head/middle/tail only, explicitly.
            scope="full_text" if label=="top20" or len(r["content"])<1100 else "head_middle_tail_only"
            limitation="助手非盲评，未独立核实事实或取得人工确认"
            if scope!="full_text":limitation+="；长文本仅抽查头中尾，未逐字核验全部内容"
            if "短非英语" in judgement:limitation+="；英语评级及长度效用可能不适合多语言消歧页"
            if identity=="BkiUgB_xK7kjXMEc_2sI":
                node=ET.fromstring(r["content"])
                observations+=f" XML根节点={node.tag}，实际节点数={sum(1 for _ in node.iter())}。"
            if identity=="BkiUeHY5qoTBG9o643V5":limitation+="；未编译或运行代码"
            rows.append({"set":label,"rank":rank,"id":identity,"domain":r["domain"],"Q":r["Q"],
                "content_characters":len(r["content"]),"content_sha256":hashlib.sha256(r["content"].encode("utf-8")).hexdigest(),
                "reviewer":annotations["reviewer"],"review_scope":scope,"observations":observations,
                "judgement":judgement,"limitations":limitation,"independent_human_confirmation":"not_obtained"})
    if len(rows)!=40 or {r["id"] for r in rows}!=set(comments):raise ValueError("Annotation/selection identity mismatch")
    write_rows(folder/"A1_text_review.csv",rows)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(folder/"quality_scoring.log",mode="a",encoding="utf-8")])
    logging.info("Text review completed: 40 selected real records annotated by assistant; independent human confirmation not obtained")
    logging.info("STOP: PHASE 4B part 1 complete; no conflict resolution, no domain allocation")
    print("Saved 40 actual assistant text reviews with exact IDs, scope and hashes; no human approval claimed")


if __name__=="__main__":main()
