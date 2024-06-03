- Location remapping (binary fields) (no rate/unweighted)
  - All fields (whole map of fields)
  - Fields (single field map)
  

- Normalized rate remapping (relative rate, only EMD)
  - Global (whole map)
  - All fields (whole map of fields)
  - Fields (single field map)


- Unnormalized rate remapping (raw rates) 
  - Global (whole map)
    - EMD (distance + rate) 
    - Rate only
  - All fields (whole map of fields)
    - EMD (distance + rate) 
    - Rate only
  - Fields (single field map)
    - EMD (distance + rate) 
    - Rate only


Unnormalized ratemap:
  - Compute + scale occupancy sample density estimates by time interval for each sample = occupancy map
  - Compute spike event density estimates + use raw
  - Divide spike event density by occupancy time density (occ map) = estimated rate of firing (unnormalized ratemap)
