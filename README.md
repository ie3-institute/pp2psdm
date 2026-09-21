# pp2psdm 


This is a work-in-progress tool for converting [pandapower](https://github.com/e2nIEE/pandapower) to [PSDM](https://github.com/ie3-institute/PowerSystemDataModel) grid models.


## Limitations 

- Currently only basic grid model conversion supported
- No switch conversion implemented yet


## Usage

```py
from pp2psdm.grid import convert_grid
import pandapower as pp

pp_grid = pp.create_empty_network()     # loaded Pandapower net
psdm_grid = convert_grid(pp_grid)
```
