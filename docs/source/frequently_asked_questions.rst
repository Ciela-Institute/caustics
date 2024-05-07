FAQs - Frequently asked questions
=================================
| **Q:** How do I know what order to put the parameter values in the pytorch tensor which gets passed to the simulator?
| **A:** If you are using a simulator, then you can get the parameters using ``your_simulator.state_dict()``. The parameters whose values are set dynamically will say "None", while the static parameters will have their values shown. The order of the dynamical parameters corresponds to the order you should use in your parameter value tensor.
|
| **Q:** Why can I put the lens redshift at higher values than the source redshift or to negative values for some parametric models?
| **A:** We can calculate everything for those profiles with reduced deflection angles where the redshifts do not actually play into the calculation. If you use a profile defined by the lens mass, like a NFW lens, or a Multiplane lens then it does matter that the redshifts make sense and you will very likely get errors for those. Similarly, if you call the ``lens.physical_deflection_angle`` you will encounter errors.
|
| **Q:** I do (multiplane-)lensing with pixelated convergence using the pixelated kappa map of a parametric profile. The lensing effect differs from directly using the parametric lens. Why is the lensing effect different?
| **A:** Since you do pixelated convergence your mass is binned in pixels in a finite field of view (FOV) so you are missing some mass. At the limit of infinite resolution and infinite FOV the pixelated profile gives you the parametric profile. If the difference is above your error tolerance then you have to increase the resolution and/or FOV of your pixelated convergence map. Especially for SIE or EPL profiles (which go to infinity density in the center, and have infinite mass outside any FOV) you will miss infinite mass when pixelating.
