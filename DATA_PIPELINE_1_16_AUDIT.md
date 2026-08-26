# MLB-BATTERS DATA PIPELINE 1–16 — 2026 CONTRACT

**Scope:** data provenance, matching, refresh, fallback, persistence and side-consistency only. **No HRR projection coefficient/formula change. No UI change.**

## 1. Current 2026 batter data
Primary current source is Baseball Savant current-season data plus MLB Stats API game logs/splits. Verified-current Savant requires MLBAM/player ID, current season when a season field exists, and current Statcast schema. Historical/career tables are not allowed to satisfy the current-Savant slot.

## 2. Current 2026 pitcher data
The same verified-current routing now applies to pitchers. Exact-pitcher Statcast pitch profiles remain live/current by MLBAM ID and are persisted only as current-season last-good fallbacks.

## 3. Historical-data isolation
`cleaned_batting_stats.csv` and `data/batter_profiles.csv` are historical/prior resources. They are **not** valid verified-current Savant data and may not replace a current 2026 table. Existing model use of a historical profile as a prior is left unchanged pending formula approval.

## 4. MLBAM / name matching
MLBAM ID is canonical. Current batter Savant matching is exact-ID first. Normalized exact name is secondary fallback only; loose fuzzy substitution is not used for the verified Savant join.

## 5. Startup verifier
Railway runs a data-pipeline verifier after all data-only patches. It checks current-route markers, MLBAM matching, historical rejection, live split/pitch/bullpen routes, fallback policy, side-probability consistency, and verified manifest/hash integrity when a promoted dataset is present.

## 6. Nightly refresh
Core batter + pitcher current-season Savant tables refresh through GitHub Actions. Downloads go to staging, both must validate, then they promote together. A failed refresh leaves current and last_good untouched.

## 7. Validation
Core files require row-count floors, MLBAM/player ID coverage, unique IDs, player names, current season, xBA/xwOBA/xSLG, HardHit%, Barrel%, Whiff%, K%, BB%, and broad Statcast metric coverage. SHA-256 hashes and source/fetched timestamps are stored in the manifest.

## 8. Timing
Nightly workflow is `30 6 * * *` UTC = 10:30 PM PST / 11:30 PM PDT, always after 10 PM Pacific.

## 9. Daytime deploy protection
Automatic supporting-data commits are scheduled only at night. Exact matchup data (splits, pitch mix, batter-vs-pitch type, bullpen workload) is refreshed live at slate time instead of creating daytime GitHub commits.

## 10. Saved official slate persistence
Runtime storage now prefers `MLB_STORAGE_DIR`, then `<RAILWAY_VOLUME_MOUNT_PATH>/mlb_engine`, then local fallback. A Railway persistent volume must actually be mounted for container-independent persistence; the repo cannot create the Railway volume itself.

## 11. Pitcher current Savant
Verified-current pitcher Savant no longer falls through to historical/career tables. Priority is fresh verified current -> live Savant -> stale verified current -> verified last_good -> missing/neutral.

## 12. Pitch arsenal freshness
Exact pitcher pitch mix is pulled by pitcher MLBAM ID from Baseball Savant Statcast search. Exact batter pitch-type performance is pulled by batter MLBAM ID. Successful live results are cached in persistent storage; only current-season last-good data can be used after a live failure. The existing formula controls remain unchanged.

## 13. Provenance / transparency
Data sidecars stamp source/state on live/cached matchup payloads and write bootstrap/verification status JSON. This creates an auditable distinction between LIVE CURRENT, VERIFIED CURRENT, LAST_GOOD, and missing/fallback data without changing cards/UI.

## 14. OVER / UNDER consistency
Production HRR remains unchanged. The verifier requires the production relationship: model over probability is calculated, pick is OVER when over probability >= 50% and UNDER otherwise, and displayed side-win probability uses `P(OVER)` for OVER and `1-P(OVER)` for UNDER. Batter Upside must also invert over probability for an UNDER side.

## 15. Bad pitcher vs UNDER sanity
A weak opposing pitcher is a positive hitter input, but it is not allowed to force every hitter OVER. Line difficulty, batter skill/split, PA, lineup slot, pitch mix, bullpen, team environment and recent evidence can still produce an UNDER. Pipeline verification focuses on proving those pitcher/split/pitch inputs are current and matched; weights are not changed here.

## 16. Final data-only operating order
1. Resolve player/team/opposing starter IDs from the current slate.
2. Pull current MLB season/log/lineup context.
3. Load verified current batter/pitcher Savant, or verified safe fallback.
4. Pull batter and pitcher handedness splits live by MLBAM ID; persist current-season last-good.
5. Pull exact starter pitch profile and exact batter-vs-pitch-type data by MLBAM ID; persist current-season last-good.
6. Pull real recent bullpen workload from MLB schedule/boxscores; persist only a short last-good window.
7. Compute existing HRR projection with the existing formula.
8. Save untouched pregame board to persistent storage.
9. Grade exactly the saved line/side after final games.

## Field-use classification

| Field family | HRR status | Current source / guard |
|---|---|---|
| Season PA/AVG/OBP/SLG/H/R/RBI/HR/BB/K | **USED IN FORMULA** | MLB Stats API current season + logs |
| Batter vs RHP/LHP splits | **USED IN FORMULA** | MLB Stats API by MLBAM ID; live-first persistent cache |
| Batter xBA/xwOBA/xSLG | **USED IN FORMULA / composite support** | Verified current Baseball Savant |
| Batter EV/HardHit%/Barrel%/Whiff%/K%/BB% | **USED IN FORMULA / composite support** | Verified current Baseball Savant / exact Statcast context |
| SweetSpot% / Zone Contact% | **LOADED; used by contact-quality/support paths where consumed** | Verified current Savant; never zero-filled when missing |
| Generic Contact% | **USED where exact pitch/contact context supplies it; otherwise fallback/missing** | Exact batter Statcast pitch-type data |
| Recent 15 / recent 30 Statcast | **USED IN EXISTING RECENT/QUALITY CONTEXT** | Current Statcast pitch-by-pitch windows |
| L3/L5 game form | **USED IN CURRENT RECENT-FORM/SELECTION SUPPORT; core HRR historical blend unchanged** | MLB game logs |
| Lineup slot / expected PA | **USED IN FORMULA** | Current lineup + MLB logs / existing PA model |
| Opposing starter identity/hand | **USED IN FORMULA** | MLB schedule probable starter + MLBAM ID |
| Pitcher ERA/WHIP/BAA/FIP/xFIP/SIERA/K% | **USED IN FORMULA** | Current MLB/pitcher contexts |
| Pitcher allowed xBA/xwOBA/xSLG/EV/HardHit/Barrel | **USED IN FORMULA / contact composite** | Current exact Statcast / verified current pitcher table |
| Pitcher vs LHB/RHB split | **USED IN FORMULA** | MLB Stats API by pitcher MLBAM ID; live-first persistent cache |
| Pitch mix / arsenal | **USED IN FORMULA** | Exact pitcher Statcast by MLBAM ID; current-season last-good fallback |
| Batter vs exact pitch types | **USED IN FORMULA** | Exact batter Statcast by MLBAM ID; current-season last-good fallback |
| Starter -> bullpen transition/leash | **USED IN FORMULA** | Existing starter leash + bullpen transition logic |
| Bullpen workload/quality | **USED IN FORMULA** | Real recent MLB boxscores + existing bullpen contexts; short last-good cache |
| Historical batter profile files | **HISTORICAL PRIOR ONLY — NOT CURRENT DATA** | Explicitly excluded from verified-current routing |

### No-manual-upload rule
Normal production operation should not require the user to upload Savant, platoon, pitch-mix, batter-vs-pitch, or bullpen files. Live/current routes are primary. Nightly verified current batter/pitcher tables and persistent last-good caches provide fallback. Manual installer support may remain in the UI as an optional emergency/research tool, but it is not required for normal operation.
