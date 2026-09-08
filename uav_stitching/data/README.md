# UAV_Graph

## Dataset layout

Raw input images and DJI positioning files are not included. The default configuration expects this layout:

```text
UAV_Graph/
  image/
    DJI_*.JPG
    information.MRK
    information.RTK
    information.NAV
    information.OBS
  uav_stitching/
    configs/
      my_uav.yaml
```

JPG/JPEG images and the MRK file are required for the metadata stage. RTK, NAV, and OBS files are optional auxiliary inputs; set their configuration entries to `null` if they are unavailable.

Update the paths in `configs/my_uav.yaml` to use another dataset location. Relative paths are resolved from `uav_stitching/`; for example, `../image` refers to `UAV_Graph/image/`. Match image files to MRK records by exposure identifier, not by directory listing or row order.
