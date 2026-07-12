# metricprobe dashboard

**Generated at:** 2025-07-24T06:00:00+00:00 · **Run:** `demo-0001` · **Git:** `demo0000demo` · **Tool:** 0.1.0

**Analysed window:** 2023-07-24 → 2025-07-24 · **as-of:** 2025-07-24

**Next update expected by:** 2025-07-28 12:00 UTC

Legend: ✅ green · ⚠️ amber · 🔴 red · ❓ indeterminate · ⏳ insufficient history · ➖ skipped. p95 = mean ± std days for a month to reach 95% of its final rows (across mature months; "> cap" when censored past lag_cap_days).

## demo_finance

| Table | Probe | Healthy? | Updating? | Complete back to | p95 (days) | p95 (months) |
| --- | --- | :---: | :---: | --- | --- | --- |
| main.settlements | settlements | ✅ | ✅ | 2025-06-12 | 40 ± 1 d | 1.3 mo |
| main.card_disputes | card_disputes | 🔴 | ✅ | 2025-06-12 | 39 ± 1 d | 1.3 mo |

## demo_health

| Table | Probe | Healthy? | Updating? | Complete back to | p95 (days) | p95 (months) |
| --- | --- | :---: | :---: | --- | --- | --- |
| main.episodes | episodes | ✅ | ✅ | 2025-06-08 | 41 ± 2 d | 1.4 mo |
| main.registry | registry | ✅ | ✅ | 2025-06-11 | 38 ± 2 d | 1.3 mo |

## demo_retail

| Table | Probe | Healthy? | Updating? | Complete back to | p95 (days) | p95 (months) |
| --- | --- | :---: | :---: | --- | --- | --- |
| main.orders | orders | ✅ | ✅ | 2025-07-03 | 19 ± 1 d | 0.6 mo |
| main.returns | returns | 🔴 | ✅ | 2025-07-02 | 19 ± 1 d | 0.6 mo |
| main.shipments | shipments | ❓ | ✅ | 2025-07-09 | 13 ± 1 d | 0.4 mo |
| main.shipments_replica | shipments_replica | ✅ | ✅ | 2025-07-09 | 13 ± 1 d | 0.4 mo |
| main.decommissioned_feed | decommissioned_feed | ➖ | ➖ | — | — | — |

## demo_sensors

| Table | Probe | Healthy? | Updating? | Complete back to | p95 (days) | p95 (months) |
| --- | --- | :---: | :---: | --- | --- | --- |
| main.telemetry | telemetry | ✅ | ✅ | 2025-07-20 | 3 ± 0 d | 0.1 mo |
| main.device_pings | device_pings | 🔴 | 🔴 | 2025-07-20 | 3 ± 0 d | 0.1 mo |
| main.gateway_logs | gateway_logs | ⚠️ | ✅ | 2025-07-20 | 3 ± 0 d | 0.1 mo |
| main.new_feed | new_feed | ⏳ | ✅ | — | — | — |

Full interactive report: [report.html](report.html) (download to open).

### orders

![orders volume](img/orders_volume.svg)
![orders completion_curves](img/orders_completion_curves.svg)
![orders completion_heatmap](img/orders_completion_heatmap.svg)
![orders percentiles](img/orders_percentiles.svg)
![orders dual_overlay](img/orders_dual_overlay.svg)
![orders dual_delta](img/orders_dual_delta.svg)

### returns

![returns volume](img/returns_volume.svg)
![returns completion_curves](img/returns_completion_curves.svg)
![returns completion_heatmap](img/returns_completion_heatmap.svg)
![returns percentiles](img/returns_percentiles.svg)
![returns dual_overlay](img/returns_dual_overlay.svg)
![returns dual_delta](img/returns_dual_delta.svg)

### shipments

![shipments volume](img/shipments_volume.svg)
![shipments completion_curves](img/shipments_completion_curves.svg)
![shipments completion_heatmap](img/shipments_completion_heatmap.svg)
![shipments percentiles](img/shipments_percentiles.svg)

### shipments_replica

![shipments_replica volume](img/shipments_replica_volume.svg)
![shipments_replica completion_curves](img/shipments_replica_completion_curves.svg)
![shipments_replica completion_heatmap](img/shipments_replica_completion_heatmap.svg)
![shipments_replica percentiles](img/shipments_replica_percentiles.svg)

### decommissioned_feed


### telemetry

![telemetry volume](img/telemetry_volume.svg)
![telemetry completion_curves](img/telemetry_completion_curves.svg)
![telemetry completion_heatmap](img/telemetry_completion_heatmap.svg)
![telemetry percentiles](img/telemetry_percentiles.svg)

### device_pings

![device_pings volume](img/device_pings_volume.svg)
![device_pings completion_curves](img/device_pings_completion_curves.svg)
![device_pings completion_heatmap](img/device_pings_completion_heatmap.svg)
![device_pings percentiles](img/device_pings_percentiles.svg)

### gateway_logs

![gateway_logs volume](img/gateway_logs_volume.svg)
![gateway_logs completion_curves](img/gateway_logs_completion_curves.svg)
![gateway_logs completion_heatmap](img/gateway_logs_completion_heatmap.svg)
![gateway_logs percentiles](img/gateway_logs_percentiles.svg)

### new_feed

![new_feed volume](img/new_feed_volume.svg)
![new_feed completion_curves](img/new_feed_completion_curves.svg)
![new_feed completion_heatmap](img/new_feed_completion_heatmap.svg)
![new_feed percentiles](img/new_feed_percentiles.svg)

### settlements

![settlements volume](img/settlements_volume.svg)
![settlements completion_curves](img/settlements_completion_curves.svg)
![settlements completion_heatmap](img/settlements_completion_heatmap.svg)
![settlements percentiles](img/settlements_percentiles.svg)
![settlements batch](img/settlements_batch.svg)

### card_disputes

![card_disputes volume](img/card_disputes_volume.svg)
![card_disputes completion_curves](img/card_disputes_completion_curves.svg)
![card_disputes completion_heatmap](img/card_disputes_completion_heatmap.svg)
![card_disputes percentiles](img/card_disputes_percentiles.svg)
![card_disputes batch](img/card_disputes_batch.svg)

### episodes

![episodes volume](img/episodes_volume.svg)
![episodes completion_curves](img/episodes_completion_curves.svg)
![episodes completion_heatmap](img/episodes_completion_heatmap.svg)
![episodes percentiles](img/episodes_percentiles.svg)
![episodes dual_overlay](img/episodes_dual_overlay.svg)
![episodes dual_delta](img/episodes_dual_delta.svg)

### registry

![registry volume](img/registry_volume.svg)
![registry completion_curves](img/registry_completion_curves.svg)
![registry completion_heatmap](img/registry_completion_heatmap.svg)
![registry percentiles](img/registry_percentiles.svg)
![registry dual_overlay](img/registry_dual_overlay.svg)
![registry dual_delta](img/registry_dual_delta.svg)

