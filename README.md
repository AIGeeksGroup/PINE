# PINE project page

Static page for "PINE: Point Map, Instance and Normal Estimation in One Diffusion Step", built from the
[Nerfies](https://github.com/nerfies/nerfies.github.io) template.

## Preview

Run `python -m http.server 8000` in this folder and open http://localhost:8000.

## Update the content

- Text, links and BibTeX: `index.html`. Lines marked `TODO` need the real authors, links and citation.
- Paper figures: `static/images/fig/`, rendered from the paper PDFs at 2000 px (wide figures) or 1100 px.
- Comparison with baselines: the appendix samples, written by
  `python ../teaser_render/export_web_compare.py ../visualization ../figure/vis_point .`
  into `static/images/compare/` and `static/js/compare_data.js`. Run it again when the appendix samples change.
  Point maps are 3D point clouds (`<method>.bin`, 100k points each, aligned to the GT) shown with three.js from
  jsDelivr; the three panels (GT, PINE, baseline) share one camera.

## Publish

Push this folder to a GitHub repository, then set Settings > Pages > Source to "Deploy from a branch", `main`, `/ (root)`.

## License

The template is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).
Keep the link to Nerfies in the footer.
