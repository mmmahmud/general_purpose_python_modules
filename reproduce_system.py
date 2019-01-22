#!/usr/bin/env python3

"""Find tidal evolution reproducing the present state of a system."""

import scipy
from astropy import units

from stellar_evolution.library_interface import MESAInterpolator
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
        print('Setting secondary phase lag to ' + repr(phase_lag(lgq)))
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
        print('Setting primary phase lag to ' + repr(phase_lag(lgq)))
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

    #False positive
    #pylint: disable=no-member
    binary = Binary(
        primary=primary,
        secondary=secondary,
        initial_orbital_period=porb_initial.to(units.day).value,
        initial_eccentricity=initial_eccentricity,
        initial_inclination=0.0,
        disk_lock_frequency=(2.0 * scipy.pi
                             /
                             disk_lock_period.to(units.day).value),
        disk_dissipation_age=disk_dissipation_age.to(units.Gyr).value,
        secondary_formation_age=disk_dissipation_age.to(units.Gyr).value
    )
    #pylint: enable=no-member
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
    secondary.configure(
        #False positive
        #pylint: disable=no-member
        age=disk_dissipation_age.to(units.Gyr).value,
        #pylint: enable=no-member
        companion_mass=primary.mass,
        #False positive
        #pylint: disable=no-member
        semimajor=binary.semimajor(porb_initial.to(units.day).value),
        #pylint: enable=no-member
        eccentricity=initial_eccentricity,
        spin_angmom=scipy.array([0.0, 0.0]),
        inclination=initial_inclination,
        periapsis=initial_periapsis,
        locked_surface=False,
        zero_outer_inclination=True,
        zero_outer_periapsis=True
    )
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
                 primary_period,
                 disk_dissipation_age,
                 max_timestep,
                 primary_lgq,
                 secondary_lgq,
                 secondary_star=False,
                 orbital_period_tolerance=1e-6):
        """
        Get ready for the solver.

        Args:
            system:    The parameters of the system we are trying to reproduce.

            interpolator:    The stellar evolution interpolator to use, could
                also be a pair of interpolators, one to use for the primary and
                one for the secondary.

            primary_period: The period at which the primaly will initially spin.

            disk_dissipation_age:    The age at which the disk dissipates and
                the secondary forms.

            max_timestep:    The maximum timestep the evolution is allowed to
                take.

        Returns:
            None
        """

        self.target_state = Structure(
            #False positive
            #pylint: disable=no-member
            age=current_age.to(units.Gyr).value,
            Porb=system.Porb.to(units.day).value,
            Pdisk=primary_period.to(units.day).value,
            planet_formation_age=disk_dissipation_age.to(units.Gyr).value
            #pylint: enable=no-member
        )
        self.system = system
        if isinstance(interpolator, MESAInterpolator):
            self.interpolator = dict(primary=interpolator,
                                     secondary=interpolator)
        else:
            self.interpolator = dict(primary=interpolator[0],
                                     secondary=interpolator[1])
        self.configuration = dict(
            #False positive
            #pylint: disable=no-member
            disk_dissipation_age=disk_dissipation_age.to(units.Gyr).value,
            max_timestep=max_timestep.to(units.Gyr).value,
            #pylint: enable=no-member
            primary_lgQ=primary_lgq,
            secondary_lgQ=secondary_lgq,
            orbital_period_tolerance=orbital_period_tolerance
        )
        self.porb_initial = None
        self.psurf = None
        self.secondary_star = secondary_star

    def __call__(self, initial_eccentricity):
        """
        Return the discrepancy in eccentricity for the given initial value.

        An evolution is found which reproduces the present day orbital period of
        the system, starting with the given initial eccentricity and the result
        of this function is the difference between the present day eccentricity
        predicted by that evolution and the measured value supplied at
        construction through the system argument. In addition, the initial
        orbital period and (initial or final, depending on which is not
        specified) stellar spin period are stored in the
        :attr:porb_initial and :attr:psurf attributes.

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
            initial_eccentricity=initial_eccentricity,
            orbital_period_tolerance=(
                self.configuration['orbital_period_tolerance']
            )
        )
        primary = create_star(
            self.system.Mprimary,
            self.system.feh,
            self.interpolator['primary'],
            self.configuration['primary_lgQ']
        )
        if self.secondary_star:
            secondary = create_star(
                self.system.Msecondary,
                self.system.feh,
                self.interpolator['secondary'],
                self.configuration['secondary_lgQ'],
                wind_strength=0.0,
                wind_saturation_frequency=1e10,
            )
        else:
            secondary = create_planet(
                self.system.Msecondary,
                self.system.Rsecondary,
                self.configuration['secondary_lgQ']
            )
        self.porb_initial, self.psurf = find_ic(self.target_state,
                                                primary,
                                                secondary)
        #False positive
        #pylint: disable=no-member
        self.porb_initial *= units.day
        self.psurf *= units.day
        final_eccentricity = find_ic.binary.final_state().eccentricity
        #pylint: enable=no-member
        print('Final eccentricity: ' + repr(final_eccentricity))
        primary.delete()
        secondary.delete()
        find_ic.binary.delete()

        return final_eccentricity - self.system.eccentricity
#pylint: enable=too-few-public-methods

def format_evolution(binary, interpolators, secondary_star):
    """Create the final result for find_evolution given an evolved binary."""

    evolution = binary.get_evolution()
    #False positive
    #pylint: disable=no-member
    evolution.orbital_period = binary.orbital_period(evolution.semimajor)

    components_to_get = ['primary']
    if secondary_star:
        components_to_get.append('secondary')

    for component in components_to_get:

        if (
                len(interpolators[component].track_masses) == 1
                and
                len(interpolators[component].track_feh) == 1
        ):
            star_params = dict(
                mass=interpolators[component].track_masses[0],
                feh=interpolators[component].track_feh[0]
            )
        else:
            star_params = dict(
                mass=getattr(binary, component).mass,
                feh=getattr(binary, component).metallicity
            )

        for quantity_name in ['radius', 'lum', 'iconv', 'irad']:
            print('Start params: ' + repr(star_params))
            quantity = interpolators[component](
                quantity_name,
                **star_params
            )
            print(quantity_name + ' age range: ' + repr((quantity.min_age,
                                                         quantity.max_age)))
            values = scipy.full(evolution.age.shape, scipy.nan)
            valid_ages = scipy.logical_and(
                evolution.age > quantity.min_age * 2.0,
                evolution.age < quantity.max_age / 2.0
            )
            values[valid_ages] = quantity(evolution.age[valid_ages])
            setattr(evolution,
                    component + '_' + quantity_name,
                    values)
            if quantity_name in ['iconv', 'irad']:
                print(
                    'Setting ' + component + '_' + quantity_name + '_' + 'star'
                )
                print('At ages: ' + repr(evolution.age[valid_ages]))
                values[valid_ages] = getattr(
                    getattr(binary, component),
                    (
                        ('envelope' if quantity_name == 'iconv' else 'core')
                        +
                        '_inertia'
                    )
                )(evolution.age[valid_ages])
                setattr(evolution,
                        component + '_' + quantity_name + '_' + 'star',
                        values)

    return evolution

def find_evolution(system,
                   interpolator,
                   *,
                   primary_lgq=None,
                   secondary_lgq=None,
                   max_age=None,
                   initial_eccentricity=0.0,
                   orbital_period_tolerance=1e-6):
    """
    Find the evolution of the given system.

    Args:
        system:    The system parameters. Usually parsed using
            read_hatsouth_info.

        interpolator:    See interpolator argument to
            EccentricitySolverCallable.__init__().

        primary_lgq:    The log10 of the tidal quality factor to assume for the
            primary.

        secondary_lgq:    The log10 of the tidal quality factor to assume for
            the secondary.

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
    secondary_star = (system.Msecondary > 0.05 * units.M_sun)
    print('System: '+ system.format())
    if hasattr(system, 'Pprimary'):
        primary_period = system.Pprimary
    else:
        primary_period = (2.0 * scipy.pi * system.Rstar
                          /
                          system.Vsini)
    disk_dissipation_age = 0.01 * units.Gyr
    max_timestep = 1e-3 * units.Gyr
    #pylint: enable=no-member
    e_solver_callable = EccentricitySolverCallable(
        system=system,
        interpolator=interpolator,
        #False positive
        #pylint: disable=no-member
        current_age=system.age,
        #pylint: enable=no-member
        primary_period=primary_period,
        disk_dissipation_age=disk_dissipation_age,
        max_timestep=max_timestep,
        primary_lgq=primary_lgq,
        secondary_lgq=secondary_lgq,
        secondary_star=secondary_star,
        orbital_period_tolerance=orbital_period_tolerance
    )
    if initial_eccentricity == 'solve':
        initial_eccentricity = scipy.optimize.brentq(e_solver_callable,
                                                     system.eccentricity,
                                                     0.5,
                                                     xtol=1e-2,
                                                     rtol=1e-2)
    e_solver_callable(initial_eccentricity)

    primary = create_star(system.Mprimary,
                          system.feh,
                          e_solver_callable.interpolator['primary'],
                          primary_lgq)

    if secondary_star:
        secondary = create_star(system.Msecondary,
                                system.feh,
                                e_solver_callable.interpolator['secondary'],
                                secondary_lgq,
                                wind_strength=0.0,
                                wind_saturation_frequency=1e10)
    else:
        secondary = create_planet(system.Msecondary,
                                  system.Rsecondary,
                                  secondary_lgq)
    print('Final evolution based on: ')
    print('\tM1 = ' + repr(primary.mass))
    print('\tM2 = ' + repr(secondary.mass))
    print('\tPdisk = ' + repr(primary_period.to('d').value))
    print('\tPorb initial = '
          +
          repr(e_solver_callable.porb_initial.to('d').value))
    print('\teccentricity = ' + repr(initial_eccentricity))
    binary = create_system(
        primary,
        secondary,
        disk_lock_period=primary_period,
        porb_initial=e_solver_callable.porb_initial,
        disk_dissipation_age=disk_dissipation_age,
        initial_eccentricity=initial_eccentricity
    )
    binary.evolve(
        #False positive
        #pylint: disable=no-member
        (max_age or system.age).to(units.Gyr).value,
        max_timestep.to(units.Gyr).value,
        #pylint: enable=no-member
        1e-6,
        None,
        True
    )

    result = format_evolution(binary,
                              e_solver_callable.interpolator,
                              secondary_star)

    primary.delete()
    secondary.delete()
    binary.delete()

    return result
