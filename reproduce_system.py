#!/usr/bin/env python3

"""Find tidal evolution reproducing the present state of a system."""

import scipy
from astropy import units

from orbital_evolution.binary import Binary
from orbital_evolution.transformations import phase_lag
from orbital_evolution.star_interface import EvolvingStar
from orbital_evolution.planet_interface import LockedPlanet
from orbital_evolution.initial_condition_solver import InitialConditionSolver
from basic_utils import Structure

def create_planet(mass, radius, lgq=None):
    """
    Return the configured planet in the given system.

    Args:
        mass:    The mass of the planets, along with astropy units.

        radius:    The radius of the planets, along with astropy units.

    """

    planet = LockedPlanet(
        #False positive
        #pylint: disable=no-member
        mass=mass(units.M_sun).value,
        radius=radius(units.R_sun).value
        #pylint: enable=no-member
    )
    if lgq is not None:
        planet.set_dissipation(
            tidal_frequency_breaks=None,
            spin_frequency_breaks=None,
            tidal_frequency_powers=scipy.array([0.0]),
            spin_frequency_powers=scipy.array([0.0]),
            reference_phase_lag=phase_lag(lgq)
        )
    return planet

def create_star(mass,
                feh,
                interpolator,
                lgq,
                *,
                wind_strength=0.13,
                wind_saturation_frequency=2.78,
                diff_rot_coupling_timescale=1.0e-2):
    """
    Create the star to use in the evolution.

    Args:
        mass:    The mass of the star to create, along with astropy units.

        feh:    The [Fe/H] value of the star to create.

        interpolator:    POET stellar evolution interpolator giving the
            evolution of the star's properties.

        lgq:    Decimal log of the tidal quality factor.

    Returns:
        EvolvingStar:
            The star in the system useable for calculating obital evolution.
    """

    #False positive
    #pylint: disable=no-member
    star = EvolvingStar(mass=mass.to(units.M_sun).value,
                        metallicity=feh,
                        wind_strength=wind_strength,
                        wind_saturation_frequency=wind_saturation_frequency,
                        diff_rot_coupling_timescale=diff_rot_coupling_timescale,
                        interpolator=interpolator)
    #pylint: enable=no-member
    print('Core formation age = ' + repr(star.core_formation_age()))
    star.select_interpolation_region(star.core_formation_age())
    if lgq is not None:
        star.set_dissipation(zone_index=0,
                             tidal_frequency_breaks=None,
                             spin_frequency_breaks=None,
                             tidal_frequency_powers=scipy.array([0.0]),
                             spin_frequency_powers=scipy.array([0.0]),
                             reference_phase_lag=phase_lag(lgq))
    return star

def create_system(primary,
                  secondary,
                  *,
                  disk_lock_period,
                  porb_initial,
                  disk_dissipation_age,
                  initial_eccentricity=0.0):
    """
    Create the system to evolve from the two bodies (primary & secondary).

    Args:
        primary:    The primary in the system. Usually created by calling
            create_star().

        planet:    The secondary in the system. Usually created by calling
            create_star() or create_planet().

        disk_lock_period:    The period to which the stellar spin is locked
            until the disk dissipates.

        porb_initial:    Initial orbital period.

        disk_dissipation_age:    The age at which the disk dissipates.

        initial_eccentricity:    The initial eccentricity of the system.

    Returns:
        Binary:
            The binary system ready to evolve.
    """

    binary = Binary(primary=primary,
                    secondary=secondary,
                    initial_orbital_period=porb_initial,
                    initial_eccentricity=initial_eccentricity,
                    initial_inclination=0.0,
                    disk_lock_frequency=(2.0 * scipy.pi / disk_lock_period),
                    disk_dissipation_age=disk_dissipation_age,
                    secondary_formation_age=disk_dissipation_age)
    binary.configure(age=primary.core_formation_age(),
                     semimajor=float('nan'),
                     eccentricity=float('nan'),
                     spin_angmom=scipy.array([0.0]),
                     inclination=None,
                     periapsis=None,
                     evolution_mode='LOCKED_SURFACE_SPIN')
    if isinstance(secondary, EvolvingStar):
        initial_inclination = scipy.array([0.0])
        initial_periapsis = scipy.array([0.0])
    else:
        initial_inclination = None
        initial_periapsis = None
    secondary.configure(age=disk_dissipation_age,
                        companion_mass=primary.mass,
                        semimajor=binary.semimajor(porb_initial),
                        eccentricity=initial_eccentricity,
                        spin_angmom=scipy.array([0.0]),
                        inclination=initial_inclination,
                        periapsis=initial_periapsis,
                        locked_surface=False,
                        zero_outer_inclination=True,
                        zero_outer_periapsis=True)
    primary.detect_stellar_wind_saturation()
    if isinstance(secondary, EvolvingStar):
        secondary.detect_stellar_wind_saturation()
    return binary

#This does serve the purpose of a single function to pass to a solver.
#pylint: disable=too-few-public-methods
class EccentricitySolverCallable:
    """Callable to pass to a solver to find the initial eccentricity."""

    def __init__(self,
                 system,
                 interpolator,
                 *,
                 current_age,
                 star_period,
                 disk_dissipation_age,
                 max_timestep,
                 primary_lgq,
                 secondary_lgq):
        """
        Get ready for the solver.

        Args:
            system:    The parameters of the system we are trying to reproduce.

            interpolator:    The stellar evolution interpolator to use.

            star_period: The period at which the star will initially spin.

            disk_dissipation_age:    The age at which the disk dissipates and
                the secondary forms.

            max_timestep:    The maximum timestep the evolution is allowed to
                take.

        Returns:
            None
        """

        self.target_state = Structure(age=current_age,
                                      Porb=system.Porb,
                                      Pdisk=star_period)
        self.system = system
        self.interpolator = interpolator
        self.configuration = dict(
            disk_dissipation_age=disk_dissipation_age,
            max_timestep=max_timestep,
            primary_lgQ=primary_lgq,
            secondary_lgQ=secondary_lgq
        )
        self.porb_initial = None
        self.psurf_initial = None

    def __call__(self, initial_eccentricity):
        """
        Return the discrepancy in eccentricity for the given initial value.

        An evolution is found which reproduces the present day orbital period of
        the system, starting with the given initial eccentricity and the result
        of this function is the difference between the present day eccentricity
        predicted by that evolution and the measured value supplied at
        construction through the system argument. In addition, the initial
        orbital period and stellar spin period are stored in the
        :attr:porb_initial and :attr:psurf_initial attributes.

        Args:
            initial_eccentricity(float):    The initial eccentricity with which
                the secondary forms.

        Returns:
            float:
                The difference between the predicted and measured values of the
                eccentricity.
        """

        find_ic = InitialConditionSolver(
            disk_dissipation_age=self.configuration['disk_dissipation_age'],
            evolution_max_time_step=self.configuration['max_timestep'],
            initial_eccentricity=initial_eccentricity
        )
        star = create_star(self.system.Mprimary,
                           self.system.feh,
                           self.interpolator,
                           self.configuration['primary_lgQ'])
        planet = create_planet(self.system,
                               self.configuration['secondary_lgQ'])
        self.porb_initial, self.psurf_initial = find_ic(self.target_state,
                                                        star,
                                                        planet)
        #False positive
        #pylint: disable=no-member
        final_eccentricity = find_ic.binary.final_state().eccentricity
        #pylint: enable=no-member
        print('Final eccentricity: ' + repr(final_eccentricity))
        star.delete()
        planet.delete()
        find_ic.binary.delete()

        return final_eccentricity - self.system.eccentricity
#pylint: enable=too-few-public-methods

def find_evolution(system,
                   interpolator,
                   *,
                   primary_lgq=None,
                   secondary_lgq=None,
                   max_age=None,
                   initial_eccentricity=0.0):
    """
    Find the evolution of the given system.

    Args:
        system:    The system parameters. Usually parsed using
            read_hatsouth_info.

        interpolator:    A stellar evolution interpolator to use. Usually from
            get_interpolator.

        primary_lgq:    The log10 of the tidal quality factor to assume for the
            primary.

        primary_lgq:    The log10 of the tidal quality factor to assume for the
            secondary.

        current_age:    The current system age at which the system parameters
            are known.

        max_age:    The maximum age up to which to calculate the evolution. If
            not specified, defaults to 1.1 * current_age.

        initial_eccentricity:    The initial eccentricity to star the evolution
            with. If set to the string 'solve' an attempt is made to find an
            initial eccentricity to reproduce the present day value given in the
            system.

    Returns:
        A structured numpy array with fields named for the quantities
        evolved.
    """


    #False positive
    #pylint: disable=no-member
    star_period = (2.0 * scipy.pi * system.Rstar
                   /
                   system.Vsini).to(units.day).value
    #pylint: enable=no-member
    disk_dissipation_age = 0.01
    max_timestep = 1e-3
    e_solver_callable = EccentricitySolverCallable(
        system=system,
        interpolator=interpolator,
        current_age=system.age,
        star_period=star_period,
        disk_dissipation_age=disk_dissipation_age,
        max_timestep=max_timestep,
        primary_lgq=primary_lgq,
        secondary_lgq=secondary_lgq
    )
    if initial_eccentricity == 'solve':
        initial_eccentricity = scipy.optimize.brentq(e_solver_callable,
                                                     system.eccentricity,
                                                     0.5,
                                                     xtol=1e-2,
                                                     rtol=1e-2)
    e_solver_callable(initial_eccentricity)

    star = create_star(system.Mprimary, system.feh, interpolator, primary_lgq)
    planet = create_planet(system.Msecondary, system.Rsecondary, secondary_lgq)
    binary = create_system(star,
                           planet,
                           disk_lock_period=e_solver_callable.psurf_initial,
                           porb_initial=e_solver_callable.porb_initial,
                           disk_dissipation_age=disk_dissipation_age,
                           initial_eccentricity=initial_eccentricity)
    binary.evolve(
        #False positive
        #pylint: disable=no-member
        max_age or 1.1 * system.age.to(units.Gyr).value,
        #pylint: enable=no-member
        max_timestep,
        1e-6,
        None
    )

    evolution_quantities = ['age',
                            'semimajor',
                            'eccentricity',
                            'envelope_angmom',
                            'core_angmom',
                            'wind_saturation']
    evolution = binary.get_evolution(evolution_quantities)
    #False positive
    #pylint: disable=no-member
    evolution.orbital_period = binary.orbital_period(evolution.semimajor)
    evolution.rstar = interpolator('radius',
                                   system.Mstar,
                                   system.FeH)(evolution.age)
    evolution.luminosity = interpolator('lum',
                                        system.Mstar,
                                        system.FeH)(evolution.age)
    #pylint: enable=no-member
    star.delete()
    planet.delete()
    binary.delete()
    result_quantities = ['age',
                         'semimajor',
                         'eccentricity',
                         'orbital_period',
                         'luminosity',
                         'rstar']
    #False positive
    #pylint: disable=no-member
    result = scipy.empty(len(evolution.age),
                         dtype=[(q, 'f8') for q in result_quantities])
    #pylint: enable=no-member
    for quantity in result_quantities:
        result[quantity] = getattr(evolution, quantity)
    print(
        'Found evolution for primary lg(Q) = %s, secondary lg(Q) = %s with '
        'a0 = %s'
        %
        (
            repr(primary_lgq),
            repr(secondary_lgq),
            repr(
                next(
                    filter(lambda a: not scipy.isnan(a), result['semimajor'])
                )
            )
        )
    )
    return result
