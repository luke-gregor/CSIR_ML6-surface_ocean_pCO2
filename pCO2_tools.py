"""
Common tools used to correct and calculate pCO2, fCO2 and fluxes
"""


class UnitError(Exception):
    pass


def check_units(arr, func, lim, msg):
    """
    Raise an error if the units are outside the given limit
    """
    from numpy import array, any
    arr = array(arr, ndmin=1)
    if any(func(arr, lim)):
        raise UnitError(msg)


def check_temp_K(temp_K):
    from numpy import less, greater
    msg = "Check that you're using the correct temperature units"
    check_units(temp_K, less, 271.15, msg)
    check_units(temp_K, greater, 318.15, msg)


def check_pres_atm(pres_atm):
    from numpy import greater, less
    msg = "Input pressure values are too large for the unit"
    check_units(pres_atm, greater, 1.5, msg)
    check_units(pres_atm, less, 0.5, msg)


def check_CO2_mol(CO2_mol):
    from numpy import greater, less
    msg = "Input CO2 is probably not in the correct unit (too large)."
    check_units(CO2_mol, greater, 0.01, msg)
    check_units(CO2_mol, less, 150e-6, msg)


def check_salt(salt):
    from numpy import greater, less
    msg = "Salinity is outside the bounds of expected (5, 50)"
    check_units(salt, less, 5, msg)
    check_units(salt, greater, 50, msg)


def check_wind_ms(wind_ms):
    from numpy import greater, less
    msg = "Salinity is outside the bounds of expected (5, 50)"
    check_units(wind_ms, less, 0, msg)
    check_units(wind_ms, greater, 40, msg)


def virial_coeff(temp_K, pres_atm, xCO2_mol=None):
    from numpy import exp

    """
    Calculate the ideal gas correction factor for converting pCO2 to fCO2.
    fCO2 = pCO2 * virial_expansion
    pCO2 = fCO2 / virial_expansion

    Based on the Lewis and Wallace 1998 Correction.

    Parameters
    ----------
    press_hPa : np.array
        uncorrected pressure in hPa
    temp_K : np.array
        temperature in degrees Kelvin
    xCO2_mol : np.array
        mole fraction of CO2. Can be pCO2/fCO2 if xCO2 is not defined or can
        leave this as undefined as makes only a small impact on output

    Return
    ------
    virial_expression : np.array
        the factor to multiply with pCO2. Unitless

    Examples
    --------
    The example below is from Dickson et al. (2007)
    >>> 350 * virial_coeff(298.15, 1)  # CO2 [uatm] * correction factor
    348.8836492182758

    References
    ----------
    Weiss, R. (1974). Carbon dioxide in water and seawater: the solubility of a
        non-ideal gas. Marine Chemistry, 2(3), 203–215.
        https://doi.org/10.1016/0304-4203(74)90015-2
    Compared with the Seacarb package in R
    """
    from numpy import array, exp

    T = array(temp_K)
    P = array(pres_atm)
    C = array(xCO2_mol)
    R = 82.057  # gas constant for ATM

    check_temp_K(T)
    check_pres_atm(P)

    # B is the virial coefficient for pure CO2
    B = -1636.75 + 12.0408 * T - 0.0327957 * T**2 + 3.16528e-5 * T**3
    # d is the virial coefficient for CO2 in air
    d = (57.7 - 0.118 * T)

    # "x2" term often neglected (assumed = 1) in applications of Weiss's (1974) equation 9
    if xCO2_mol is not None:
        check_CO2_mol(C)
        x2 = (1 - C)**2
    else:
        x2 = 1

    ve = exp(P * (B + 2 * x2 * d) / (R * T))

    return ve


def vapress_weiss1980(salt, temp_K):
    """
    Calculates the water vapour pressure of seawater at a given salinity and
    temerature using the methods defined in Weiss (1974)

    Parameters
    ----------
    salt : np.array
        salinity
    temp_K : np.array
        temperature in deg Kelvin

    Returns
    -------
    sea_vapress : np.array
        sea water vapour pressure in atm

    Examples
    --------
    >>> vapress_weiss1980(35, 25+273.15)  # tempC + 273.15
    0.03065529996317971

    """
    from numpy import array, exp, log

    T = array(temp_K)
    S = array(salt)

    check_temp_K(T)
    check_salt(S)

    # Equation comes straight from Weiss and Price (1980)
    pH2O = exp(+ 24.4543
               - 67.4509 * (100 / T)
               - 4.8489 * log(T / 100)
               - 0.000544 * S)

    return pH2O


def vapress_dickson2007(salt, temp_K):
    """
    Calculates the water vapour pressure of seawater at a given salinity and
    temerature using the methods defined in Dickson et al. (2007; CO2 manual)

    Parameters
    ----------
    salt : np.array
        salinity
    temp_K : np.array
        temperature in deg Kelvin

    Returns
    -------
    sea_vapress : np.array
        sea water vapour pressure in atm

    Examples
    --------
    >>> vapress_dickson2007(35, 298.15)  # from Dickson et al. (2007) Ch 5.3.2
    0.030698866245809465

    """
    from numpy import array, exp, log

    T = array(temp_K)
    S = array(salt)

    check_temp_K(T)

    ###################################################
    # WATER VAPOUR PRESSURE FOR PURE WATER
    ###################################################
    # alpha coefficients from Wafner and Pruss, (2002)
    a1 = -7.85951783
    a2 = +1.84408259
    a3 = -11.7866497
    a4 = +22.6807411
    a5 = -15.9618719
    a6 = +1.80122502
    # critical points for water
    Pc = 22.064 / 101325.0e-6  # convert to atmosphers
    Tc = 647.096
    # zeta numbers correspond with alpha numbers
    z = 1 - T / Tc
    z1 = z
    z2 = z**1.5
    z3 = z**3
    z4 = z**3.5
    z5 = z**4
    z6 = z**7.5
    # vapour pressure of pure water
    pure_water = Pc * exp((Tc / T) * (a1*z1 + a2*z2 + a3*z3 + a4*z4 + a5*z5 + a6*z6))

    ###################################################
    # WATER VAPOUR PRESSURE FOR SEA WATER
    ###################################################
    # osmotic coeffcients at 25C - Millero 1974
    c0 = +0.90799
    c1 = -0.08992
    c2 = +0.18458
    c3 = -0.07395
    c4 = -0.00221
    # total molality of dissolved species
    total_molality = 31.998 * S / (1e3 - 1.005*S)
    B1 = total_molality * 0.5
    B2 = B1**2
    B3 = B1**3
    B4 = B1**4
    osmotic_coeff = c0 + c1*B1 + c2*B2 + c3*B3 + c4*B4

    seawater = pure_water * exp(-0.018 * osmotic_coeff * total_molality)

    return seawater


def temperature_correction(temp_in, temp_out):
    """
    Calculate a correction factor for the temperature difference between the
    intake and equilibrator. This is based on the empirical relationship used
    in Takahashi et al. 1993.
    pCO2_Tout = pCO2_Tin * T_factor

    Parameters
    ----------
    temp_in : np.array
        temperature at which original pCO2 is measured
    temp_out : np.array
        temperature for which pCO2 should be represented

    Return
    ------
    factor : np.array
        a correction factor to be multiplied to pCO2 (unitless)

    References
    ----------
    Takahashi, Taro et al. (1993). Seasonal variation of CO2 and nutrients in
        the high-latitude surface oceans: A comparative study. Global
        Biogeochemical Cycles, 7(4), 843–878. https://doi.org/10.1029/93GB02263
    """

    from numpy import array, exp
    # see the Takahashi 1993 paper for full description

    Ti = array(temp_in)
    To = array(temp_out)

    factor = exp(0.0433 * (To - Ti) - 4.35E-05 * (To**2 - Ti**2))

    return factor


def pressure_height_correction(pres_hPa, tempSW_C, sensor_height=10.):
    """
    Returns exact sea level pressure if the sensor is measuring at height

    Parameters
    ----------
    pres_hPa : np.array
        Pressure in kiloPascal measured at height
    temp_C : np.array
        Temperature of the seawater in deg C
    sensor_height : float
        the height of the sensor above sea level. Can be negative if you want
        to convert SLP to sensor height pressure

    Return
    ------
    presCor_kPa : np.array
        height corrected pressure
    """
    from numpy import array

    P = array(pres_hPa) * 100  # pressure in Pascal
    T = array(tempSW_C) + 273.15  # temperature in Kelvin

    check_temp_K(T)
    check_pres_atm(P / 101325)

    # Correction for pressure based on sensor height
    R = 8.314  # universal gas constant (J/mol/K)
    M = 0.02897  # molar mass of air in (kg/mol) - Wikipedia
    # Density of air at a given temperature. Here we assume
    # that the air temp is the same as the intake temperature
    d = P / (R / M * T)
    g = 9.8  # gravity in (m/s2)
    h = -sensor_height  # height in (m)
    # correction for atmospheric
    press_height_corr_hpa = (P - (d * g * h)) / 100.

    return press_height_corr_hpa


def fCO2_to_pCO2(fCO2SW_uatm, tempSW_C, pres_hPa=None, tempEQ_C=None):
    """
    Convert fCO2 to pCO2 for SOCAT in sea water. A simple version of the
    equation would simply be:
        pCO2sw = fCO2sw / virial_exp
    where the virial expansion is calculated without xCO2

    We get a simple approximate for equilibrator xCO2 with:
        xCO2eq = fCO2sw * deltaTemp(sw - eq) / press_eq

    pCO2sw is then calculated with:
        pCO2sw = fCO2sw / virial_exp(xCO2eq)

    Parameters
    ----------
    fCO2SW_uatm : array
        seawater fugacity of CO2 in micro atmospheres
    tempSW_C : array
        sea water temperature in degrees C/K
    tempEQ_C : array
        equilibrator temperature in degrees C/K
    pres_hPa : array
        equilibrator pressure in kilo Pascals

    Returns
    -------
    pCO2SW_uatm : array
        partial pressure of CO2 in seawater

    Note
    ----
    In FluxEngine, they account for the change in xCO2. This error is so small
    that it is not significant to be concerned about it. Their correction is
    more precise, but the difference between their iterative correction and our
    approximation is on the order of 1e-14 atm (or 1e-8 uatm).

    Examples
    --------
    >>> fCO2_to_pCO2(380, 8)
    381.50806485658234
    >>> fCO2_to_pCO2(380, 8, pres_hPa=985)
    381.4659553134281
    >>> fCO2_to_pCO2(380, 8, pres_hPa=985, tempEQ_C=14)
    381.466027968504
    """

    from numpy import abs, array, any

    # if equilibrator inputs are None then make defaults Patm=1, tempEQ=tempSW
    if tempEQ_C is None:
        tempEQ_C = tempSW_C
    if pres_hPa is None:
        pres_hPa = 1013.25

    # standardise the inputs and convert units
    fCO2sw = array(fCO2SW_uatm) * 1e-6
    Tsw = array(tempSW_C) + 273.15
    Teq = array(tempEQ_C) + 273.15
    Peq = array(pres_hPa) / 1013.25

    # check if units make sense
    check_pres_atm(Peq)
    check_CO2_mol(fCO2sw)
    check_temp_K(Tsw)
    check_temp_K(Teq)

    # calculate the CO2 difference due to equilibrator and seawater temperatures
    dT = temperature_correction(Tsw, Teq)
    # a best estimate of xCO2 - this is an aproximation
    # one would have to use pCO2 / Peq to get real xCO2
    xCO2eq = fCO2sw * dT / Peq

    pCO2SW = fCO2sw / virial_coeff(Tsw, Peq, xCO2eq)
    pCO2SW_uatm = pCO2SW * 1e6

    return pCO2SW_uatm


def pCO2_to_fCO2(pCO2SW_uatm, tempSW_C, pres_hPa=None, tempEQ_C=None):
    """
    Convert fCO2 to pCO2 for SOCAT in sea water. A simple version of the
    equation would simply be:
        fCO2sw = pCO2sw / virial_exp
    where the virial expansion is calculated without xCO2

    We get a simple approximate for equilibrator xCO2 with:
        xCO2eq = pCO2sw * deltaTemp(sw - eq) / press_eq

    fCO2sw is then calculated with:
        fCO2sw = pCO2sw * virial_exp(xCO2eq)

    Parameters
    ----------
    pCO2SW_uatm : array
        seawater fugacity of CO2 in micro atmospheres
    tempSW_C : array
        sea water temperature in degrees C/K
    tempEQ_C : array
        equilibrator temperature in degrees C/K
    pres_hPa : array
        pressure in kilo Pascals

    Returns
    -------
    fCO2SW_uatm : array
        partial pressure of CO2 in seawater

    Note
    ----
    In FluxEngine, they account for the change in xCO2. This error is so small
    that it is not significant to be concerned about it. Their correction is
    more precise, but the difference between their iterative correction and our
    approximation is less than 1e-14 atm (or 1e-8 uatm).

    Examples
    --------
    >>> pCO2_to_fCO2(380, 8)
    378.49789637942064
    >>> pCO2_to_fCO2(380, 8, pres_hPa=985)
    378.53967828231225
    >>> pCO2_to_fCO2(380, 8, pres_hPa=985, tempEQ_C=14)
    378.53960618459695
    """
    from numpy import abs, array, any

    # if equilibrator inputs are None then make defaults Patm=1, tempEQ=tempSW
    if tempEQ_C is None:
        tempEQ_C = tempSW_C
    if pres_hPa is None:
        pres_hPa = 1013.25

    # standardise the inputs and convert units
    pCO2sw = array(pCO2SW_uatm) * 1e-6
    Tsw = array(tempSW_C) + 273.15
    Teq = array(tempEQ_C) + 273.15
    Peq = array(pres_hPa) / 1013.25

    # check if units make sense
    check_pres_atm(Peq)
    check_CO2_mol(pCO2sw)
    check_temp_K(Tsw)
    check_temp_K(Teq)

    # calculate the CO2 difference due to equilibrator and seawater temperatures
    dT = temperature_correction(Tsw, Teq)
    # a best estimate of xCO2 - this is an aproximation
    # one would have to use pCO2 / Peq to get real xCO2
    xCO2eq = pCO2sw * dT / Peq

    fCO2sw = pCO2sw * virial_coeff(Tsw, Peq, xCO2eq)
    fCO2sw_uatm = fCO2sw * 1e6

    return fCO2sw_uatm


def solubility_weiss1974(salt, temp_K, press_atm=1):
    """
    Calculates the solubility of CO2 in sea water for the calculation of
    air-sea CO2 fluxes. We use the formulation by Weiss (1974) summarised in
    Wanninkhof (2014).

    Parameters
    ----------
    salt : np.array
        salinity in PSU
    temp_K : np.array
        temperature in deg Kelvin
    press_atm : np.array
        pressure in atmospheres. Used in the solubility correction for water
        vapour pressure. If not given, assumed that press_atm is 1atm

    Returns
    -------
    K0 : np.array
        solubility of CO2 in seawater in mol/L/atm

    Examples
    --------
    >>> solubility_weiss1974(35, 299.15)  # from Weiss (1974) Table 2 but with pH2O correction
    0.029285284543519093
    """

    from numpy import array, exp, log

    T = array(temp_K)
    S = array(salt)
    P = array(press_atm)

    check_temp_K(T)
    check_salt(S)
    check_pres_atm(P)

    # from table in Wanninkhof 2014
    a1 = -58.0931
    a2 = +90.5069
    a3 = +22.2940
    b1 = +0.027766
    b2 = -0.025888
    b3 = +0.0050578

    T100 = T / 100
    K0 = exp(a1 +
             a2*(100/T) +
             a3*log(T100) +
             S*(b1 + b2*T100 + b3*T100**2))

    pH2O = vapress_weiss1980(S, T)
    K0 = (K0 / (P - pH2O))

    return K0  # units mol/L/atm


def solubility_woolf2016(salt, temp_K, deltaT, press_atm=1):
    """
    A wrapper around solubility calculated using the Weiss (1974) approach.
    This is taken from the FluxEngine script.

    Parameters
    ----------
    temp_K : np.array
        temperature of sea water at the desired level (e.g. skin)
    salt : np.array
        salinity of seawater
    deltaT : np.array
        SST differences (foundation - skin)

    Returns
    -------
    K0 : np.array
        solubility of CO2 in seawater in mol/L/atm
    """
    K0 = solubility_weiss1974(salt, temp_K, press_atm)

    return K0 * (1 - 0.015*deltaT)


def schmidt_number(temp_C):
    """
    Calculates the Schmidt number as defined by Jahne et al. (1987) and listed
    in Wanninkhof (2014) Table 1.

    Parameters
    ----------
    temp_C : np.array
        temperature in degrees C

    Returns
    -------
    Sc : np.array
        Schmidt number (dimensionless)

    Examples
    --------
    >>> schmidt_number(20)  # from Wanninkhof (2014)
    668.344

    """

    from numpy import array

    T = array(temp_C)
    check_temp_K(T + 273.15)

    a = +2116.8
    b = -136.25
    c = +4.7353
    d = -0.092307
    e = +0.0007555

    Sc = a + b*T + c*T**2 + d*T**3 + e*T**4

    return Sc


class gas_transfer:

    @staticmethod
    def k_Li86(wind_ms, temp_C):
        """
        Calculates the gas transfer coeffcient for CO2 using the formulation
        of Liss and Merlivat (1986)

        Parameters
        ----------
        wind_ms : np.array
            wind speed in m/s
        temp_C : np.array
            temperature in degrees C

        Returns
        -------
        kw : array
            gas transfer velocity (k600) in cm/hr
        """
        from numpy import array, zeros_like

        U = array(wind_ms)
        T = array(temp_C)

        check_temp_K(T + 273.15)
        check_wind_ms(U)

        Sc = schmidt_number(T)
        k = zeros_like(temp_C)

        i1 = U <= 3.6
        i2 = (U > 3.6) & (U < 13.)
        i3 = U >= 13.

        k[i1] = (0.17 * U[i1]) * (Sc[i1] / 600)**(-2. / 3.)
        k[i2] = ((U[i2] - 3.4) * 2.8) * (600 / Sc[i2])**0.5
        k[i3] = ((U[i3] - 8.4) * 5.9) * (600 / Sc[i3])**0.5
        return k

    @staticmethod
    def k_Wa92(wind_ms, temp_C):
        """
        Calculates the gas transfer coeffcient for CO2 using the formulation
        of Wanninkhof (1992)
            k660 = 0.39 * u^2

        Parameters
        ----------
        wind_ms : np.array
            wind speed in m/s
        temp_C : np.array
            temperature in degrees C

        Returns
        -------
        kw : array
            gas transfer velocity (k660) in cm/hr
        """
        from numpy import array

        U = array(wind_ms)
        T = array(temp_C)

        check_temp_K(T + 273.15)
        check_wind_ms(U)

        Sc = schmidt_number(temp_C)

        return (0.39 * U**2) * (660 / Sc)**0.5

    @staticmethod
    def k_Sw07(wind_ms, temp_C):
        """
        Calculates the gas transfer coeffcient for CO2 using the formulation
        of Sweeny et al (2007) who scaled Wanninkhof (1992)
            k660 = 0.27 * u^2

        Parameters
        ----------
        wind_ms : np.array
            wind speed in m/s
        temp_C : np.array
            temperature in degrees C

        Returns
        -------
        kw : array
            gas transfer velocity (k660) in cm/hr
        """
        from numpy import array

        U = array(wind_ms)
        T = array(temp_C)

        check_temp_K(T + 273.15)
        check_wind_ms(U)

        Sc = schmidt_number(temp_C)

        return (0.27 * U**2) * (660 / Sc)**0.5

    @staticmethod
    def k_Wa99(wind_ms, temp_C):
        """
        Calculates the gas transfer coeffcient for CO2 using the formulation
        of Wanninkhof (1999)
            k600 = 0.0283 * U^3

        Parameters
        ----------
        wind_ms : np.array
            wind speed in m/s
        temp_C : np.array
            temperature in degrees C

        Returns
        -------
        kw : array
            gas transfer velocity (k600) in cm/hr
        """
        from numpy import array

        U = array(wind_ms)
        T = array(temp_C)

        check_temp_K(T + 273.15)
        check_wind_ms(U)

        Sc = schmidt_number(temp_C)

        return (0.0283 * U**3) * (600 / Sc)**0.5

    @staticmethod
    def k_Ni00(wind_ms, temp_C):
        """
        Calculates the gas transfer coeffcient for CO2 using the formulation
        of Nightingale et al (2000)
            k600 = (0.333 * U) + (0.222 * U^2)

        Parameters
        ----------
        wind_ms : np.array
            wind speed in m/s
        temp_C : np.array
            temperature in degrees C

        Returns
        -------
        kw : array
            gas transfer velocity (k600) in cm/hr
        """
        from numpy import array

        U = array(wind_ms)
        T = array(temp_C)

        check_temp_K(T + 273.15)
        check_wind_ms(U)

        Sc = schmidt_number(temp_C)

        return (0.333 * U + 0.222 * U**2) * (600 / Sc)**0.5

    @staticmethod
    def k_Ho06(wind_ms, temp_C):
        """
        Calculates the gas transfer coeffcient for CO2 using the formulation
        of Ho et al (2006)
            k600 = 0.266 * U^2

        Parameters
        ----------
        wind_ms : np.array
            wind speed in m/s
        temp_C : np.array
            temperature in degrees C

        Returns
        -------
        kw : array
            gas transfer velocity (k600) in cm/hr
        """
        from numpy import array

        U = array(wind_ms)
        T = array(temp_C)

        check_temp_K(T + 273.15)
        check_wind_ms(U)

        Sc = schmidt_number(temp_C)

        return (0.266 * U**2) * (600 / Sc)**0.5

    @staticmethod
    def k_Wa09(wind_ms, temp_C):
        """
        Calculates the gas transfer coeffcient for CO2 using the formulation
        of Wanninkhof et al. (2009)
            k660 = 3. + (0.1 * U) + (0.064 * U^2) + (0.011 * U^3)

        Parameters
        ----------
        wind_ms : np.array
            wind speed in m/s
        temp_C : np.array
            temperature in degrees C

        Returns
        -------
        kw : array
            gas transfer velocity (k660) in cm/hr
        """
        from numpy import array

        U = array(wind_ms)
        T = array(temp_C)

        check_temp_K(T + 273.15)
        check_wind_ms(U)

        Sc = schmidt_number(temp_C)

        return (3. + 0.1*U + 0.064*U**2 + 0.011*U**3) * (660 / Sc)**0.5

    @staticmethod
    def k_Mc01(wind_ms, temp_C):
        """
        Calculates the gas transfer coeffcient for CO2 using the formulation
        of McGillis et al. (2001)
            k660 = 3.3 + (0.026 * U^3)

        Parameters
        ----------
        wind_ms : np.array
            wind speed in m/s
        temp_C : np.array
            temperature in degrees C

        Returns
        -------
        kw : array
            gas transfer velocity (k660) in cm/hr
        """
        from numpy import array

        U = array(wind_ms)
        T = array(temp_C)

        check_temp_K(T + 273.15)
        check_wind_ms(U)

        Sc = schmidt_number(temp_C)

        return 3.3 + (0.026 * U**3) * (660/Sc)**0.5


def flux_woolf2016_rapid(temp_bulk_C, salt_bulk, pCO2_bulk_uatm, pCO2_air_uatm,
                         press_hPa, wind_ms,
                         kw_func=gas_transfer.k_Ho06,
                         cool_skin_bias=0.14, salty_skin_bias=0.1):
    """
    Calculates air sea CO2 fluxes using the RAPID model as defined by Woolf et
    al. (2016), where the concentration of CO2 in the skin and foundation layers
    are used to calculate the fluxes rather than delta pCO2 (latter is called
    bulk flux).

    We calculate the skin temperature and salinity using a cool and salty skin
    bias as defined in Woolf et al. (2016). The defaults are 0.14 degC and
    0.1 PSU as taken from FluxEngine.

    **Assumptions: ** This function is set up to use AVHRR only OISST which
    reports temperatures at 1m depth based on a buoy correction (Banzon et al.
    2016). We make the assumption that this bulk temperature is equivalent to
    foundation temperature (where nighttime and daytime temperatures are the
    same). We also assume that EN4 salinity is foundation salinity (this is
    probably more accurate than the first assumtion). Lastly we assume that the
    ML estimated fCO2 is bulk fCO2 as we use bulk variable inputs (SSS and SST).

    Parameters
    ----------
    temp_bulk_C : np.array
        temperature from OISST in deg Celcius with an allowable range of
        [-2 : 45]
    salt_bulk : np.array
        salinity from EN4 in PSU. Allowable range [5 : 50]
    pCO2_bulk_uatm : np.array
        partial pressure of CO2 in the sea in micro-atmospheres, assuming that
        it was measured/predicted at the same level as the temperature and
        salinity (See our assumptions above). Allowable range is [50 : 1000]
    pCO2_air_uatm : np.array
        partial pressure of CO2 in the air in micro-atmospheres. Allowable range
        is [50 : 1000]. We recommend the use of CarboScope atmospheric pCO2 by
        Rodenbeck et al. (2014) at http://www.bgc-jena.mpg.de/CarboScope/
    press_hPa : np.array
        atmospheric pressure in hecto-Pascals with an allowable range of
        [500 : 1500] hPa
    wind_ms : np.array
        wind speed in metres per second with an allowable range of [0 : 40]
    kw_func : callable
        a function that returns the gas transfer velocity in cm/hr. The default
        is the gas transfer volicty as calculated by Ho et al. (2006). This
        is the prefered method of Goddijn-Murphy et al. (2016). Other functions
        are available in the `gas_transfer` class. If you'd like to use your own
        inputs must be wind speed (m/s) and temperature (degC) and output must
        be cm/hr
    cool_skin_bias : float
        The temperature difference between the foundation/bulk temperature and the
        skin temperature as suggested by Wolf et al. (2016). The default is
        0.14 degC where this will be subtracted from the bulk temperature, i.e.
        the surface is cooler due to the cooling effect of winds.
    salty_skin_bias : float
        The salinity difference between the foundation and skin layers. This is
        driven by evaporation and defaults to 0.1 (will be added to salinity).

    Reurns
    ------
    FCO2 : np.array
        Sea-air CO2 flux where positive is out of the ocean and negative is into
        the ocean. Units are gC.m-2.day-1 (grams Carbon per metre squared per day)
    """
    from seawater import dens0
    from numpy import array

    press_atm = array(press_hPa) / 1013.25

    SSTfnd_C = array(temp_bulk_C)
    SSTskn_C = SSTfnd_C - cool_skin_bias  # from default FluxEngine config
    SSTfnd_K = SSTfnd_C + 273.15
    SSTskn_K = SSTskn_C + 273.15
    SSTdelta = SSTfnd_C - SSTskn_C

    SSSfnd = array(salt_bulk)
    SSSskn = SSSfnd + salty_skin_bias  # from default FluxEngine config

    pCO2sea = array(pCO2_bulk_uatm) * 1e-6  # to atm
    pCO2air = array(pCO2_air_uatm) * 1e-6

    # checking units
    check_temp_K(SSTfnd_K)
    check_salt(SSSfnd)
    check_pres_atm(press_atm)
    check_CO2_mol(pCO2sea)
    check_CO2_mol(pCO2air)
    check_wind_ms(wind_ms)

    fCO2sea = pCO2sea * virial_coeff(SSTfnd_K, press_atm)
    fCO2air = pCO2air * virial_coeff(SSTskn_K, press_atm)

    # units in mol . L-1 . atm-1
    K0fnd = solubility_woolf2016(SSSfnd, SSTfnd_K, SSTdelta, press_atm)
    K0skn = solubility_woolf2016(SSSskn, SSTskn_K, SSTdelta, press_atm)

    # molar mass of carbon (gC/mol * kg/g)
    mC = 12.0108 * 1000  # kg . mol-1

    # CONC : UNIT ANALYSIS
    #         solubility         *  pCO2 *  molar mass
    # conc = (mol . L-1 . atm-1) * (atm) * (kg . mol-1)
    # conc = mol. mol-1 . L-1 . atm . atm-1 * kg
    # conc = kg . L-1    |||    gC . m-3
    # Bulk uses skin, equilibrium and rapid use foundation for concSEA
    concSEA = K0fnd * fCO2sea * mC
    concAIR = K0skn * fCO2air * mC

    # KW : UNIT ANALYSIS
    # kw = (cm / 100) / (hr / 24)
    # kw = m . day-1
    kw = gas_transfer.k_Ho06(wind_ms, SSTskn_C) * (24 / 100)

    # FLUX : UNIT ANALYSIS
    # flux = (m . day-1) * (g . m-3)
    # flux = gC . m . m-3 . day-1
    # flux = gC . m-2 . day-1
    CO2flux_woolfe = kw * (concSEA - concAIR)

    return CO2flux_woolfe


def flux_bulk(temp_bulk_C, salt_bulk, pCO2_bulk_uatm, pCO2_air_uatm,
              press_hPa, wind_ms, kw_func=gas_transfer.k_Ho06):
    """
    Calculates bulk air-sea CO2 fluxes: FCO2 = kw * K0 * dfCO2, without defining
    skin and foundation concentration differences as in the RAPID model.

    Parameters
    ----------
    temp_bulk_C : np.array
        temperature from OISST in deg Celcius with an allowable range of [-2 : 45]
    salt_bulk : np.array
        salinity from EN4 in PSU. Allowable range [5 : 50]
    pCO2_bulk_uatm : np.array
        partial pressure of CO2 in the sea in micro-atmospheres. Allowable range
        is [50 : 1000]
    pCO2_air_uatm : np.array
        partial pressure of CO2 in the air in micro-atmospheres. Allowable range
        is [50 : 1000]. We recommend the use of CarboScope atmospheric pCO2 by
        Rodenbeck et al. (2014) at http://www.bgc-jena.mpg.de/CarboScope/
    press_hPa : np.array
        atmospheric pressure in hecto-Pascals with an allowable range of [500 : 1500]
    wind_ms : np.array
        wind speed in metres per second with an allowable range of [0 : 40]
    kw_func : callable
        a function that returns the gas transfer velocity in cm/hr. The default
        is the gas transfer volicty as calculated by Ho et al. (2006). This
        is the prefered method of Goddijn-Murphy et al. (2016). Other functions
        are available in the `gas_transfer` class. If you'd like to use your own
        inputs must be wind speed (m/s) and temperature (degC) and output must
        be cm/hr

    Reurns
    ------
    FCO2 : np.array
        Sea-air CO2 flux where positive is out of the ocean and negative is into
        the ocean. Units are gC.m-2.day-1 (grams Carbon per metre squared per day)
    """
    from seawater import dens0
    from numpy import array

    press_atm = array(press_hPa) / 1013.25

    SSTfnd_C = array(temp_bulk_C)
    SSTfnd_K = SSTfnd_C + 273.15

    SSSfnd = array(salt_bulk)

    pCO2sea = array(pCO2_bulk_uatm) * 1e-6  # to atm
    pCO2air = array(pCO2_air_uatm) * 1e-6

    # checking units
    check_temp_K(SSTfnd_K)
    check_salt(SSSfnd)
    check_pres_atm(press_atm)
    check_CO2_mol(pCO2sea)
    check_CO2_mol(pCO2air)
    check_wind_ms(wind_ms)

    fCO2sea = pCO2sea * virial_coeff(SSTfnd_K, press_atm)
    fCO2air = pCO2air * virial_coeff(SSTfnd_K, press_atm)

    K0blk = solubility_weiss1974(SSSfnd, SSTfnd_K, press_atm)

    # molar mas of carbon in g . mmol-1
    mC = 12.0108 * 1000  # (g . mol-1) / (mmol . mol-1)

    # KW : UNIT ANALYSIS
    # kw = (cm . hr-1) * hr . day-1 . cm-1 . m
    # kw = m . day-1
    kw = gas_transfer.k_Ho06(wind_ms, SSTfnd_C) * (24 / 100)

    # flux = (m . day-1) .  (mol . L-1 . atm-1) . atm . (gC . mmol-1)
    # flux = (m . day-1) . (mmol . m-3 . atm-1) . atm . (gC . mmol-1)
    # flux = gC . m-2 . day-1
    CO2flux_bulk = kw * K0blk * (fCO2sea - fCO2air) * mC

    return CO2flux_bulk


def test_flux():
    """
    For doctest.testmod(). Compares the rapid and bulk formulations of sea-air
    CO2 fluxes where fluxes into the ocean should be bigger for rapid and fluxes
    out of the ocean for rapid should be smaller than bulk.

    Tests
    -----
    Seaward fluxes should be larger for woolf
    >>> seaward_woolf = flux_woolf2016_rapid(25, 35, 300, 400, 1013.25, 20)
    >>> seaward_bulk = flux_bulk(25, 35, 300, 400, 1013.25, 20)
    >>> seaward_woolf < seaward_bulk  # seaward flux is negative
    True

    Source fluxes should be smaller for woolf
    >>> seaward_woolf = flux_woolf2016_rapid(25, 35, 400, 300, 1013.25, 20)
    >>> seaward_bulk = flux_bulk(25, 35, 400, 300, 1013.25, 20)
    >>> seaward_woolf < seaward_bulk  # source flux is positive
    True


    """


if __name__ == "__main__":
    from doctest import testmod

    testmod(verbose=True)
