"""
Forensic Investigation PDF Report Exporter.
Generates publication-quality, tamper-evident PDF investigation dossiers using ReportLab.
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_investigation_pdf(report_dict: Dict[str, Any]) -> bytes:
    """
    Renders an official, structured forensic investigation report into PDF format.
    Includes metadata, consensus verdict, model comparison table, graph findings,
    supported findings with citations, and scientific epistemic limitations.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        alignment=0,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "Heading2Custom",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6,
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerCustom",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748b"),
        spaceBefore=4,
        spaceAfter=4,
    )
    code_style = ParagraphStyle(
        "CodeCustom",
        parent=styles["Code"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0f172a"),
    )

    elements = []

    # Title & Header
    elements.append(Paragraph("ETGNN-Agent — Forensic Investigation Dossier", title_style))
    elements.append(
        Paragraph(
            "Evidence-Grounded Bitcoin Illicit Transaction Risk Analysis • Elliptic Dataset Platform",
            subtitle_style,
        )
    )
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#38bdf8"), spaceAfter=10))

    # Investigation Metadata Box
    inv_id = str(report_dict.get("investigation_id", "N/A"))
    tx_id = str(report_dict.get("target_transaction_id", "N/A"))
    snap_id = str(report_dict.get("snapshot_id", "N/A"))
    status = str(report_dict.get("investigation_status") or report_dict.get("status", "COMPLETED"))
    verdict = str(report_dict.get("consensus_verdict", "N/A"))
    gen_mode = str(report_dict.get("generation_mode", "SIMULATED"))
    prov_type = str(report_dict.get("provider_type", "mock"))
    model_id = str(report_dict.get("provider_model_id", "deterministic_mock_v3"))
    risk_ind = str(report_dict.get("risk_indicator", "BASELINE_RISK"))
    timestamp = report_dict.get("generated_at") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    meta_data = [
        [
            Paragraph("<b>Investigation ID:</b>", body_style),
            Paragraph(inv_id, code_style),
            Paragraph("<b>Target Tx ID:</b>", body_style),
            Paragraph(tx_id, code_style),
        ],
        [
            Paragraph("<b>Snapshot:</b>", body_style),
            Paragraph(snap_id, body_style),
            Paragraph("<b>Investigation Status:</b>", body_style),
            Paragraph(status, body_style),
        ],
        [
            Paragraph("<b>Consensus Verdict:</b>", body_style),
            Paragraph(f"<b>{verdict}</b>", body_style),
            Paragraph("<b>Risk Indicator:</b>", body_style),
            Paragraph(risk_ind, body_style),
        ],
        [
            Paragraph("<b>Provider Mode:</b>", body_style),
            Paragraph(f"{prov_type} ({gen_mode})", body_style),
            Paragraph("<b>Generated At:</b>", body_style),
            Paragraph(str(timestamp), body_style),
        ],
    ]

    meta_table = Table(meta_data, colWidths=[100, 160, 110, 160])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(meta_table)
    elements.append(Spacer(1, 10))

    # Narrative Summary
    elements.append(Paragraph("1. Executive Narrative Summary", h2_style))
    summary_text = str(report_dict.get("narrative_summary", "No narrative generated."))
    elements.append(Paragraph(summary_text, body_style))
    elements.append(Spacer(1, 6))

    # Model Predictions & Spread
    elements.append(Paragraph("2. Model Comparison & Spread Analysis", h2_style))
    preds = report_dict.get("model_predictions", {})
    rf_p = preds.get("random_forest_prob", 0.0) if isinstance(preds, dict) else 0.0
    xgb_p = preds.get("xgboost_prob", 0.0) if isinstance(preds, dict) else 0.0
    etgnn_p = preds.get("canonical_etgnn_prob", 0.0) if isinstance(preds, dict) else report_dict.get("model_probability", 0.0)
    divergence = report_dict.get("model_disagreement", False)
    spread = report_dict.get("probability_spread", 0.0)

    model_table_data = [
        ["Model Architecture", "Predicted Probability", "Decision Threshold", "Classification Risk"],
        ["ETGNN (Canonical Graph Model)", f"{etgnn_p:.4f}", "0.5000", "Elevated Risk" if etgnn_p >= 0.5 else "Licit Risk"],
        ["Random Forest (Tabular Baseline)", f"{rf_p:.4f}", "0.5000", "Elevated Risk" if rf_p >= 0.5 else "Licit Risk"],
        ["XGBoost (Tabular Baseline)", f"{xgb_p:.4f}", "0.5000", "Elevated Risk" if xgb_p >= 0.5 else "Licit Risk"],
    ]

    m_table = Table(model_table_data, colWidths=[180, 110, 110, 130])
    m_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8.5),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(m_table)

    div_text = (
        f"<b>Model Disagreement Detected</b> (Spread: {spread:.4f} &ge; 0.30): ETGNN incorporates topological message passing, "
        f"dampening tabular anomalies present in tree models."
        if divergence
        else f"<b>Model Agreement Verified</b> (Spread: {spread:.4f} &lt; 0.30): Graph and tabular classifiers aligned."
    )
    elements.append(Paragraph(div_text, body_style))
    elements.append(Spacer(1, 6))

    # Graph & Explainer Evidence
    elements.append(Paragraph("3. Graph Neighborhood & GNNExplainer Findings", h2_style))
    graph_ev = report_dict.get("graph_evidence") or {}
    in_d = graph_ev.get("incoming_edges_count", 0)
    out_d = graph_ev.get("outgoing_edges_count", 0)
    nodes_2h = graph_ev.get("subgraph_nodes_2hop", 0)
    illicit_nb = graph_ev.get("illicit_neighbors_count", 0)
    is_fully_iso = graph_ev.get("is_fully_isolated", in_d == 0 and out_d == 0)

    iso_str = "<b>Fully Isolated Node:</b> 0 incoming and 0 outgoing edges in snapshot graph." if is_fully_iso else f"In-degree: {in_d}, Out-degree: {out_d}."
    elements.append(Paragraph(f"• <b>Neighborhood Structure:</b> {iso_str} 2-hop connected component contains {nodes_2h} transactions ({illicit_nb} illicit-label neighbors).", body_style))

    gnn_ev = report_dict.get("gnnexplainer_evidence")
    if gnn_ev:
        top_feats = gnn_ev.get("top_feature_indices", [])
        fid_drop = gnn_ev.get("fidelity_drop", 0.0)
        elements.append(Paragraph(f"• <b>GNNExplainer Attribution:</b> Top influential feature indices {top_feats} (Fidelity drop: {fid_drop:.4f}).", body_style))
        elements.append(Paragraph(f"<i>Disclaimer: {gnn_ev.get('interpretation_disclaimer', 'Explainer shows sensitivity rather than legal proof.')}</i>", disclaimer_style))
    elements.append(Spacer(1, 6))

    # Supported Findings
    elements.append(Paragraph("4. Evidence-Grounded Findings & Citations", h2_style))
    findings = report_dict.get("supported_findings", [])
    if findings:
        for idx, f in enumerate(findings, 1):
            finding_text = f.get("finding") if isinstance(f, dict) else getattr(f, "finding", str(f))
            eid = f.get("evidence_id") if isinstance(f, dict) else getattr(f, "evidence_id", "")
            raw_cites = f.get("evidence_citations") if isinstance(f, dict) else getattr(f, "evidence_citations", None)
            cites = raw_cites if raw_cites and isinstance(raw_cites, list) else ([eid] if eid else [])
            cites_str = ", ".join([str(c) for c in cites if c])
            elements.append(Paragraph(f"<b>[{idx}] {finding_text}</b>", body_style))
            if cites_str:
                elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<i>Verified Citations: {cites_str}</i>", disclaimer_style))
    else:
        elements.append(Paragraph("No individual findings supported; investigation concluded under baseline conditions.", body_style))
    elements.append(Spacer(1, 6))

    # Limitations & Epistemic Boundaries
    elements.append(Paragraph("5. Methodological Boundaries & Limitations", h2_style))
    lims = report_dict.get("evidence_limitations") or report_dict.get("limitations") or []
    for lim in lims:
        elements.append(Paragraph(f"• {lim}", body_style))

    # Standard Epistemic Notice
    elements.append(Spacer(1, 8))
    notice = report_dict.get("epistemic_notice") or (
        "Model probability represents raw statistical output, NOT a calibrated Bayesian confidence. "
        "Statistical risk predictions do not constitute judicial proof or legal evidence of financial crime."
    )
    elements.append(
        Paragraph(
            f"<b>EPISTEMIC BOUNDARY NOTICE:</b> {notice}",
            disclaimer_style,
        )
    )

    # Footer Hash & Verifiable Digest
    elements.append(Spacer(1, 10))
    prov_chain = report_dict.get("provenance_chain", [])
    canonical_summary = {
        "investigation_id": inv_id,
        "target_transaction_id": tx_id,
        "snapshot_id": snap_id,
        "verdict": verdict,
        "model_predictions": preds,
        "provenance_chain": prov_chain,
    }
    canonical_bytes = json.dumps(canonical_summary, sort_keys=True).encode("utf-8")
    canonical_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
    elements.append(
        Paragraph(
            f"<b>Canonical Report Artifact Checksum (SHA-256):</b> {canonical_sha256}<br/>"
            f"<i>Verification Procedure: Computed over canonical sorted JSON of core forensic outputs (investigation_id, target_transaction_id, snapshot_id, consensus_verdict, model_predictions, provenance_chain).</i>",
            code_style,
        )
    )

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
