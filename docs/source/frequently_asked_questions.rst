FAQs - Frequently asked questions
=================================

| **Q:** Why can I put the lens redshift at higher values than the source redshift or to negative values for parametric models that are defined by the Einstein radius instead of a mass?
| **A:** We can calculate everything for those profiles with reduced deflection angles where the redshifts do not actually play into the calculation. If you use a profile defined by the lens mass, like a NFW lens, or a Multiplane lens then it does matter that the redshifts make sense and you will very likely get errors for those.
|
| **Q:** I do (multiplane-)lensing with pixelated convergence using the pixelated kappa map of a parametric profile. The lensing effect differs from directly using the parametric lens. Why is the lensing effect different?
| **A:** Since you do pixelated convergence your mass is binned in pixels and you are missing some mass. At the limit of infinite resolution the pixelated profile gives you the parametric profile. So you have to increase the resolution of your pixelated convergence map. Especially for SIE or EPL profiles (going to infinity density in the center) you will miss infinite mass when pixelating.
