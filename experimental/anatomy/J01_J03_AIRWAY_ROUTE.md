# J01–J03 | Source-derived route rehearsal foundation

Existing Source Anatomy inspector only. Load exact H CT and I airway mask pair, then the airway-skeleton.json emitted by I02. No parallel application, no changes to original reference bronchoscopy or A4 gate.

J01: binds graph to CT and mask identities; checks nodes land on occupied voxel centers and each edge is local and has correct physical length. Rejects diagonal edges unless the crossed voxel block is foreground. Graph may retain cycles, and no anatomical branch names or pruning are inferred. User sets an explicit root; Dijkstra creates a rooted path, not an anatomical branch tree.

J02: select entry and target at MPR crosshair; nearest skeleton node within 5 mm is reported with its snap distance (off-lumen target possible; route terminates on skeleton, not at lesion). Choose outer instrument diameter and radial margin. The solver minimizes physical path length over edges satisfying a conservative radius lower bound. On orthogonal grids, distance from node to the nearest background voxel box (or outside volume) is computed locally and capped; subtract half edge length from the minimum endpoint distance using the 1-Lipschitz property. This intentionally rejects some feasible routes; it is a rigid spherical-envelope geometry approximation, not shaft curvature, tissue compliance, uncertainty or safe clinical passage. Sheared grids rejected. A 12,000-node search limit and per-node sample cap constrain work; offline processing may be needed for large studies. Browser graph identity is metadata, not an authenticated signed report. Mask edits invalidate route.

J03: review each planned node on source MPR with physical RAS position and forward unit tangent; export candidate JSON. No endoscopic lumen surface/camera renderer, swept shaft mechanics, procedure authorization or XR binding yet. Next J refinement must create a verified surface and frame transport before actual endoscopic rehearsal. Test with independently reviewed real airway data and measured instruments.

Run node --test experimental/tests/j03-airway-route.test.mjs. Synthetic fixtures only; no patient data committed.
