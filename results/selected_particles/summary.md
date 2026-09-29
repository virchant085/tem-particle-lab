# TEM tracking result

Method: template; frames: 26; selected IDs: [1, 2, 3, 4, 6, 7].

Observed 131/156 selected target-frames. Coverage is not accuracy. Review 25 missing/uncertain target-frames in review_required.csv.

Green circles are fixed selected ROIs, yellow paths link only consecutive observed centers. No invented coordinates on missing frames. A manual anchor can restart a lost ID.

tracks.csv keeps raw image coordinates and separately background-relative coordinates; the latter are invalid after an untrusted drift step. This estimates motion relative to the selected background, not proven stage drift.

acquisition_fps and nm_per_pixel remain null unless explicitly supplied. Video time is playback time. The embedded experimental clock/scale are not automatically interpreted. ROI area is not segmented particle area.

No formal GT metrics are asserted. See comparison outputs for explicitly scoped visual-reference checks.
