"""One page underwriting and pricing quote sheet (PDF)."""
from __future__ import annotations

import io
from datetime import date

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

NAVY = colors.HexColor("#6D1A36")  # bordeaux primary
GOLD = colors.HexColor("#B08D57")
LIGHT = colors.HexColor("#F7F2EF")


def _eur(x: float) -> str:
    return f"EUR {x:,.0f}"


def _table(rows, widths, header=True):
    t = Table(rows, colWidths=widths)
    style = [("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E3D6D0")),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def build_quote_pdf(app, pol, uw, res, author: str = "Giacomo Signorini") -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm)
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Times-Bold", fontSize=17, textColor=NAVY, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading3"], fontName="Times-Bold", fontSize=11, textColor=NAVY, spaceBefore=6, spaceAfter=3)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8, leading=10)
    small = ParagraphStyle("s", parent=body, fontSize=7, textColor=colors.grey)
    W = doc.width

    el = [Paragraph("Jumbo Life Underwriting Quote", h1),
          Paragraph(f"Illustrative case study prepared by {author} | {date.today():%d %B %Y} | "
                    "Not an offer of insurance", small), Spacer(1, 4)]

    el.append(Paragraph("Case", h2))
    el.append(_table([
        ["Entry age", str(app.age), "Sum assured", _eur(pol.sum_assured)],
        ["Pricing basis", "Unisex (EU)" if pol.unisex else f"Sex specific ({app.sex})", "Term", f"{pol.term} years"],
        ["Tobacco", "Smoker" if app.smoker else "Non-smoker", "Premium mode", pol.premium_mode.capitalize()],
        ["Purpose", app.purpose.replace("_", " ").capitalize(), "Residence risk class", str(app.residence_risk_class)],
    ], [W * 0.2, W * 0.3, W * 0.2, W * 0.3], header=False))

    el.append(Paragraph("Underwriting decision", h2))
    el.append(Paragraph(f"<b>{uw.decision}</b> | Class: <b>{uw.risk_class}</b> | Mortality debit "
                        f"{uw.total_debit:+.0f}% | Flat extras {uw.max_flat_extra_permille:.1f} per mille", body))
    rows = [["Factor", "Finding", "Debit", "Flat extra"]]
    for l in uw.lines:
        fe = "" if not l.flat_extra_permille else (
            f"{l.flat_extra_permille:.1f} per mille" + (f" for {l.flat_extra_years} yrs" if l.flat_extra_years else ""))
        rows.append([l.factor, Paragraph(l.finding, body), f"{l.debit_pct:+.0f}%" if l.debit_pct else "", fe])
    el.append(_table(rows, [W * 0.18, W * 0.47, W * 0.12, W * 0.23]))
    el.append(Spacer(1, 3))
    el.append(Paragraph("<b>Evidence required:</b> " + ", ".join(uw.evidence), body))
    for f in uw.financial_flags:
        el.append(Paragraph("<b>Financial underwriting:</b> " + f, body))

    el.append(Paragraph("Premium", h2))
    unit = "per year" if pol.premium_mode == "annual" else "single premium"
    d = res.decomposition
    el.append(_table([
        ["Gross premium", f"{_eur(res.premium)} {unit}", "Rate", f"{res.rate_permille:.3f} per mille"],
        ["PV expected claims", _eur(d["Expected claims"]), "PV expenses", _eur(d["Expenses and underwriting"])],
        ["PV commission", _eur(d["Distribution commission"]), "PV profit (gross)", _eur(d["Profit margin"])],
    ], [W * 0.2, W * 0.3, W * 0.2, W * 0.3], header=False))

    el.append(Paragraph("Reinsurance placement", h2))
    rows = [["Layer", "Type", "Rating", "Sum assured", "Share", "YRT rate"]]
    for l in res.layers:
        rows.append([l.name, l.kind, l.rating, _eur(l.sum_assured), f"{l.sum_assured / pol.sum_assured:.1%}",
                     f"{l.rate_pct:.0%} of BE" if l.rate_pct else ""])
    el.append(_table(rows, [W * 0.25, W * 0.15, W * 0.1, W * 0.2, W * 0.12, W * 0.18]))
    r = res.reinsurance_summary
    el.append(Spacer(1, 3))
    el.append(_table([
        ["Net cost of reinsurance (PV)", _eur(r["Net cost of reinsurance"]), "PV profit net", _eur(r["PV profit net"])],
        ["Premium for target margin net", _eur(r["Premium for target margin net of reinsurance"]), "", ""],
    ], [W * 0.28, W * 0.22, W * 0.2, W * 0.3], header=False))

    c = res.capital
    el.append(Paragraph("Solvency II life underwriting capital (simplified standard formula)", h2))
    el.append(_table([
        ["", "Gross", "Net of reinsurance"],
        ["SCR mortality (+15%)", _eur(c["SCR mortality gross"]), _eur(c["SCR mortality net"])],
        ["SCR catastrophe (+1.5 per mille)", _eur(c["SCR catastrophe gross"]), _eur(c["SCR catastrophe net"])],
        ["SCR life (aggregated)", _eur(c["SCR life gross"]), _eur(c["SCR life net"])],
        ["PV cost of capital", _eur(c["PV cost of capital gross"]), _eur(c["PV cost of capital net"])],
    ], [W * 0.4, W * 0.3, W * 0.3]))

    if res.alerts:
        el.append(Paragraph("Alerts", h2))
        for a in res.alerts:
            el.append(Paragraph("• " + a, body))

    el.append(Spacer(1, 6))
    el.append(Paragraph("All ratings, mortality parameters and reinsurance terms are illustrative assumptions "
                        "chosen by the author and do not reflect the basis of any insurer or reinsurer.", small))
    doc.build(el)
    return buf.getvalue()
