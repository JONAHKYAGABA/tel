#!/usr/bin/env bash
# scripts/download_rag_docs.sh
#
# Downloads PDFs and HTML pages most relevant to the 5G RAN troubleshooting
# scenarios in this challenge. Saves everything into knowledge/raw/ so
# build_kb_index.py can chunk + embed them.
#
# Sources (all free, no auth):
#   1. 3GPP TS specs (PDFs hosted on ETSI) — RRC, PHY, NR overall, measurements
#   2. arXiv open-access papers on 5G drive-test, handover failure, PDCCH
#   3. ShareTechNote HTML pages (may be region-blocked — best-effort)
#
# Usage:
#   bash scripts/download_rag_docs.sh
#   ls -la knowledge/raw/
#
# Total: ~150 MB across 20-30 files (mostly 3GPP PDFs).

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p knowledge/raw
cd knowledge/raw

c_blue()  { printf "\033[1;34m%s\033[0m\n" "$1"; }
c_green() { printf "\033[1;32m%s\033[0m\n" "$1"; }
c_yel()   { printf "\033[1;33m%s\033[0m\n" "$1"; }
c_red()   { printf "\033[1;31m%s\033[0m\n" "$1"; }

UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
OK=0
FAIL=0

try_download() {
    # $1 = url, $2 = output filename
    local url="$1" out="$2"
    if [ -f "$out" ] && [ "$(stat -c%s "$out" 2>/dev/null || echo 0)" -gt 1000 ]; then
        c_yel "  SKIP (exists $(du -h "$out" | cut -f1)): $out"
        return 0
    fi
    if curl -sL --max-time 90 --retry 2 --retry-delay 3 \
            -A "$UA" -o "$out" "$url"; then
        if [ "$(stat -c%s "$out" 2>/dev/null || echo 0)" -gt 5000 ]; then
            c_green "  OK $(du -h "$out" | cut -f1): $out"
            OK=$((OK+1))
            return 0
        fi
    fi
    c_red "  FAIL $url"
    rm -f "$out"
    FAIL=$((FAIL+1))
    return 1
}

# ============ 3GPP specs (ETSI mirror) ==============================
c_blue "=== 3GPP TS specs from ETSI (PDFs) ==="

# TS 38.331 - RRC protocol (A2/A3/A5 events, handover, reestablishment)
# Try Rel-18 then Rel-17.
try_download "https://www.etsi.org/deliver/etsi_ts/138300_138399/138331/18.02.00_60/ts_138331v180200p.pdf" "3gpp_38331_rrc.pdf" || \
try_download "https://www.etsi.org/deliver/etsi_ts/138300_138399/138331/17.07.00_60/ts_138331v170700p.pdf" "3gpp_38331_rrc.pdf"

# TS 38.300 - NR overall architecture
try_download "https://www.etsi.org/deliver/etsi_ts/138300_138399/138300/18.02.00_60/ts_138300v180200p.pdf" "3gpp_38300_nr_overall.pdf" || \
try_download "https://www.etsi.org/deliver/etsi_ts/138300_138399/138300/17.07.00_60/ts_138300v170700p.pdf" "3gpp_38300_nr_overall.pdf"

# TS 38.214 - PHY procedures (CQI/MCS, RB allocation, BLER)
try_download "https://www.etsi.org/deliver/etsi_ts/138200_138299/138214/18.02.00_60/ts_138214v180200p.pdf" "3gpp_38214_phy_proc.pdf" || \
try_download "https://www.etsi.org/deliver/etsi_ts/138200_138299/138214/17.07.00_60/ts_138214v170700p.pdf" "3gpp_38214_phy_proc.pdf"

# TS 38.215 - Layer 1 measurements (RSRP, RSRQ, SINR definitions)
try_download "https://www.etsi.org/deliver/etsi_ts/138200_138299/138215/18.02.00_60/ts_138215v180200p.pdf" "3gpp_38215_measurements.pdf" || \
try_download "https://www.etsi.org/deliver/etsi_ts/138200_138299/138215/17.07.00_60/ts_138215v170700p.pdf" "3gpp_38215_measurements.pdf"

# TS 38.133 - Measurement requirements (thresholds, accuracy)
try_download "https://www.etsi.org/deliver/etsi_ts/138100_138199/138133/18.04.00_60/ts_138133v180400p.pdf" "3gpp_38133_meas_req.pdf" || \
try_download "https://www.etsi.org/deliver/etsi_ts/138100_138199/138133/17.09.00_60/ts_138133v170900p.pdf" "3gpp_38133_meas_req.pdf"

# TS 38.213 - PHY scheduling (PDCCH, CCE, search spaces)
try_download "https://www.etsi.org/deliver/etsi_ts/138200_138299/138213/18.02.00_60/ts_138213v180200p.pdf" "3gpp_38213_phy_sched.pdf" || \
try_download "https://www.etsi.org/deliver/etsi_ts/138200_138299/138213/17.07.00_60/ts_138213v170700p.pdf" "3gpp_38213_phy_sched.pdf"

# TS 38.211 - Physical channels and modulation
try_download "https://www.etsi.org/deliver/etsi_ts/138200_138299/138211/18.02.00_60/ts_138211v180200p.pdf" "3gpp_38211_phy_channels.pdf" || \
try_download "https://www.etsi.org/deliver/etsi_ts/138200_138299/138211/17.07.00_60/ts_138211v170700p.pdf" "3gpp_38211_phy_channels.pdf"

# TS 38.321 - MAC protocol (HARQ, scheduling info)
try_download "https://www.etsi.org/deliver/etsi_ts/138300_138399/138321/18.02.00_60/ts_138321v180200p.pdf" "3gpp_38321_mac.pdf" || \
try_download "https://www.etsi.org/deliver/etsi_ts/138300_138399/138321/17.07.00_60/ts_138321v170700p.pdf" "3gpp_38321_mac.pdf"

# ============ arXiv papers ==========================================
c_blue "=== arXiv open-access papers ==="

try_download "https://arxiv.org/pdf/2308.07050"        "arxiv_2308_07050_5g_survey.pdf"
try_download "https://arxiv.org/pdf/2106.06477"        "arxiv_2106_06477_5g_drive_test_ml.pdf"
try_download "https://arxiv.org/pdf/2009.11414"        "arxiv_2009_11414_lte_handover_failure.pdf"
try_download "https://arxiv.org/pdf/2304.12365"        "arxiv_2304_12365_ran_anomaly.pdf"
try_download "https://arxiv.org/pdf/2210.05952"        "arxiv_2210_05952_5g_kpi_opt.pdf"
try_download "https://arxiv.org/pdf/2103.16273"        "arxiv_2103_16273_handover_optim.pdf"
try_download "https://arxiv.org/pdf/2304.06962"        "arxiv_2304_06962_pdcch.pdf"
try_download "https://arxiv.org/pdf/2207.05317"        "arxiv_2207_05317_self_organizing_5g.pdf"
try_download "https://arxiv.org/pdf/2206.09124"        "arxiv_2206_09124_5g_coverage_optim.pdf"
try_download "https://arxiv.org/pdf/2109.04977"        "arxiv_2109_04977_5g_beam_management.pdf"

# ============ ShareTechNote (best-effort — may be region-blocked) ===
c_blue "=== ShareTechNote HTML pages (best-effort) ==="

for stn in \
  "5G_SS_RSRP" "5G_SS_SINR" "5G_SS_RSRQ" \
  "5G_HandOver" "5G_MeasurementEvent" "5G_RRCConnectionReestablishment" \
  "5G_BeamForming" "5G_PDCCH" "5G_PDSCH" \
  "5G_NeighbourCellList" "5G_PCI" "5G_DriveTest" \
  "5G_PowerControl" "5G_RandomAccess" "5G_BLER" "5G_CSIRS" "5G_CQI"; do
    try_download "https://www.sharetechnote.com/html/5G/${stn}.html" "stn_${stn}.html"
done

# ============ Wikipedia primers (low-priority but useful) ===========
c_blue "=== Wikipedia primers ==="

try_download "https://en.wikipedia.org/wiki/5G_NR" "wiki_5g_nr.html"
try_download "https://en.wikipedia.org/wiki/Reference_signal_received_power" "wiki_rsrp.html"
try_download "https://en.wikipedia.org/wiki/Handover" "wiki_handover.html"
try_download "https://en.wikipedia.org/wiki/Physical_cell_identity" "wiki_pci.html"

# ============ summary ==============================================
echo
c_blue "==========================================="
c_green "  $OK files downloaded successfully"
[ "$FAIL" -gt 0 ] && c_red "  $FAIL files failed (check log above)"
c_blue "==========================================="
echo "Output in: $(pwd)"
du -sh .
ls -la | tail -30
