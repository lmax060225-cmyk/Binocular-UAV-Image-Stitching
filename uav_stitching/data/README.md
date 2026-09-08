# Data placement

No 491 MB image set is duplicated into this directory. The checked-in-style
configuration resolves `../image` relative to the project root, which points to
the supplied sibling directory:

```text
D:\UAV_Paper\image\
  DJI_*.JPG
  information.MRK
  information.RTK
  information.NAV
  information.OBS
```

For another dataset, change only the relative or absolute paths in
`configs/my_uav.yaml`; source code contains no dataset-specific absolute path.
