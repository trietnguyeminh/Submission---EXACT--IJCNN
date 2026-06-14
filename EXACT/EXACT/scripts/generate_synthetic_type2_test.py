#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
import re
import unicodedata
from collections.abc import Callable
from pathlib import Path


K_COULOMB = 8.99e9
EPS0 = 8.854e-12
MU0 = 4.0 * math.pi * 1e-7
G = 9.8
E_CHARGE = 1.602e-19
E_MASS = 9.109e-31


def fmt(value: float) -> str:
    if abs(value) < 1e-15:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1.0e4 or magnitude < 1.0e-2:
        exponent = math.floor(math.log10(magnitude))
        mantissa = value / (10.0**exponent)
        mantissa_text = f"{mantissa:.4f}".rstrip("0").rstrip(".")
        if abs(float(mantissa_text)) >= 10:
            mantissa_text = f"{mantissa / 10:.4f}".rstrip("0").rstrip(".")
            exponent += 1
        return f"{mantissa_text} x 10^{exponent}"
    text = f"{value:.5f}".rstrip("0").rstrip(".")
    return text if text else "0"


def normalize_question(text: str) -> str:
    value = unicodedata.normalize("NFKC", text)
    replacements = {
        "×": "x",
        "−": "-",
        "–": "-",
        "—": "-",
        "μ": "u",
        "µ": "u",
        "\r": " ",
        "\n": " ",
        "\t": " ",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


def row(
    prefix: str,
    question: str,
    cot: str,
    answer: float | str,
    unit: str,
    *,
    template_id: str,
) -> dict[str, str]:
    answer_text = answer if isinstance(answer, str) else fmt(answer)
    return {
        "id": prefix,
        "question": question,
        "cot": f"{template_id}: {cot}",
        "answer": answer_text,
        "unit": unit,
    }


def choose(rng: random.Random, values: list[float] | list[int]) -> float:
    return rng.choice(values)


def cap_energy(rng: random.Random) -> dict[str, str]:
    c_uf = choose(rng, [1, 2.2, 3.3, 4.7, 6.8, 10, 15, 22, 33, 47, 68, 100, 150, 220])
    u = choose(rng, [6, 9, 12, 18, 24, 30, 48, 60, 120, 240])
    energy = 0.5 * c_uf * 1e-6 * u**2
    q = f"A capacitor has capacitance C = {c_uf:g} uF and voltage U = {u:g} V. Calculate the stored electric energy."
    return row("TD", q, "Use W = 1/2 C U^2 after converting uF to F.", energy, "J", template_id="cap_energy")


def cap_charge(rng: random.Random) -> dict[str, str]:
    c_uf = choose(rng, [2, 5, 8, 12, 25, 40, 75, 120, 180, 330])
    u = choose(rng, [5, 10, 15, 20, 36, 50, 80, 110, 220])
    charge = c_uf * 1e-6 * u
    q = f"Find the charge on a {c_uf:g} uF capacitor when it is connected across {u:g} V."
    return row("TD", q, "Use Q = C U with C in farads.", charge, "C", template_id="cap_charge")


def cap_voltage_from_charge(rng: random.Random) -> dict[str, str]:
    c_uf = choose(rng, [3, 6, 9, 12, 18, 30, 45, 60, 90])
    q_uc = choose(rng, [12, 18, 24, 36, 48, 72, 96, 150, 240])
    voltage = q_uc / c_uf
    q = f"A capacitor of capacitance {c_uf:g} uF carries charge {q_uc:g} uC. Determine the voltage across it."
    return row("TD", q, "Use U = Q/C; uC/uF gives volts.", voltage, "V", template_id="cap_voltage_from_charge")


def cap_series(rng: random.Random) -> dict[str, str]:
    c1 = choose(rng, [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30])
    c2 = choose(rng, [3, 5, 7, 9, 12, 15, 18, 25, 35, 50])
    ceq = c1 * c2 / (c1 + c2)
    q = f"Two capacitors C1 = {c1:g} uF and C2 = {c2:g} uF are connected in series. Calculate the equivalent capacitance in uF."
    return row("TD", q, "For series capacitors, Ceq = C1 C2/(C1+C2).", ceq, "uF", template_id="cap_series")


def cap_parallel(rng: random.Random) -> dict[str, str]:
    c1 = choose(rng, [1.5, 2.5, 4, 6, 10, 15, 22, 33, 47])
    c2 = choose(rng, [2, 3.5, 5, 8, 12, 18, 27, 39, 56])
    ceq = c1 + c2
    q = f"Capacitors of {c1:g} uF and {c2:g} uF are joined in parallel. What is their total capacitance?"
    return row("TD", q, "For parallel capacitors, capacitances add directly.", ceq, "uF", template_id="cap_parallel")


def cap_disconnected_dielectric_voltage(rng: random.Random) -> dict[str, str]:
    c_uf = choose(rng, [4, 6, 10, 15, 22, 47, 100])
    u = choose(rng, [12, 24, 40, 60, 90, 150, 300])
    k = choose(rng, [2, 2.5, 3, 4, 5, 6])
    new_u = u / k
    q = (
        f"A {c_uf:g} uF capacitor is charged to {u:g} V, disconnected from the battery, "
        f"then a dielectric with relative permittivity {k:g} is inserted. Find the new voltage."
    )
    return row("TD", q, "Disconnected capacitor keeps Q constant, so U' = U/k.", new_u, "V", template_id="cap_disconnected_dielectric_voltage")


def cap_connected_dielectric_charge(rng: random.Random) -> dict[str, str]:
    c_uf = choose(rng, [2, 5, 10, 20, 50, 80, 120])
    u = choose(rng, [9, 12, 24, 36, 48, 72])
    k = choose(rng, [2, 3, 4, 5])
    charge = k * c_uf * 1e-6 * u
    q = (
        f"A capacitor C = {c_uf:g} uF remains connected to a {u:g} V source while a dielectric k = {k:g} is inserted. "
        "Calculate the final charge."
    )
    return row("TD", q, "Connected capacitor keeps U constant, so Q' = k C U.", charge, "C", template_id="cap_connected_dielectric_charge")


def cap_parallel_plate(rng: random.Random) -> dict[str, str]:
    radius_cm = choose(rng, [2, 3, 4, 5, 6, 8, 10])
    d_mm = choose(rng, [0.4, 0.6, 0.8, 1.0, 1.5, 2.0])
    er = choose(rng, [1, 2.2, 3.5, 4.0, 5.5])
    area = math.pi * (radius_cm / 100) ** 2
    capacitance = EPS0 * er * area / (d_mm / 1000)
    q = (
        f"A parallel-plate capacitor has circular plates of radius {radius_cm:g} cm separated by {d_mm:g} mm. "
        f"The dielectric constant is {er:g}. Find its capacitance."
    )
    return row("TD", q, "Use C = eps0 er A/d with A = pi r^2.", capacitance, "F", template_id="cap_parallel_plate")


def cap_breakdown_charge(rng: random.Random) -> dict[str, str]:
    area_cm2 = choose(rng, [12, 20, 35, 50, 80, 120, 180])
    emax_kv_per_mm = choose(rng, [1.5, 2, 2.5, 3, 4, 5])
    charge = EPS0 * (area_cm2 * 1e-4) * (emax_kv_per_mm * 1e6)
    q = (
        f"A parallel-plate capacitor has plate area {area_cm2:g} cm^2 in air. "
        f"If breakdown field is {emax_kv_per_mm:g} kV/mm, calculate the maximum charge before breakdown."
    )
    return row("TD", q, "At breakdown Qmax = C Umax = eps0 A Emax.", charge, "C", template_id="cap_breakdown_charge")


def coulomb_force(rng: random.Random) -> dict[str, str]:
    q1 = choose(rng, [1.2, 1.8, 2.4, 3.5, 4.8, 6.0, 7.5, 9.0])
    q2 = choose(rng, [1.5, 2.0, 2.8, 4.0, 5.5, 8.0, 10.0])
    r_cm = choose(rng, [3, 4, 5, 6, 8, 10, 12, 15, 20])
    force = K_COULOMB * q1 * 1e-6 * q2 * 1e-6 / (r_cm / 100) ** 2
    q = f"Two point charges {q1:g} uC and {q2:g} uC are {r_cm:g} cm apart in air. Find the magnitude of the electrostatic force."
    return row("LD", q, "Apply Coulomb law F = k |q1 q2|/r^2.", force, "N", template_id="coulomb_force")


def point_field(rng: random.Random) -> dict[str, str]:
    charge_uc = choose(rng, [0.8, 1.5, 2.2, 3.3, 4.7, 6.8, 8.2, 12])
    r_cm = choose(rng, [2, 3, 4, 5, 8, 10, 15, 25])
    field = K_COULOMB * charge_uc * 1e-6 / (r_cm / 100) ** 2
    q = f"What is the electric field magnitude at a point {r_cm:g} cm from a charge Q = {charge_uc:g} uC in air?"
    return row("LD", q, "Use E = k |Q|/r^2.", field, "N/C", template_id="point_field")


def force_in_uniform_field(rng: random.Random) -> dict[str, str]:
    charge_uc = choose(rng, [0.5, 1.2, 2.0, 3.6, 5.0, 7.5, 10])
    field_kv = choose(rng, [2, 4, 6, 8, 12, 20, 35])
    force = charge_uc * 1e-6 * field_kv * 1000
    q = f"A particle carrying charge {charge_uc:g} uC is placed in a uniform electric field of {field_kv:g} kV/m. Calculate the electric force magnitude."
    return row("LD", q, "Use F = qE after converting kV/m to V/m.", force, "N", template_id="force_in_uniform_field")


def perpendicular_fields(rng: random.Random) -> dict[str, str]:
    q1 = choose(rng, [1.5, 2.5, 4.0, 6.0, 8.0])
    q2 = choose(rng, [1.2, 3.0, 5.0, 7.5, 9.0])
    r1_cm = choose(rng, [3, 4, 5, 6, 8, 10])
    r2_cm = choose(rng, [4, 5, 7, 9, 12, 15])
    e1 = K_COULOMB * q1 * 1e-6 / (r1_cm / 100) ** 2
    e2 = K_COULOMB * q2 * 1e-6 / (r2_cm / 100) ** 2
    resultant = math.hypot(e1, e2)
    q = (
        f"At point M, charge q1 = {q1:g} uC is {r1_cm:g} cm away along the x direction and "
        f"charge q2 = {q2:g} uC is {r2_cm:g} cm away along the y direction. The field components are perpendicular. "
        "Find the resultant electric field magnitude at M."
    )
    return row("LD", q, "Compute E1 and E2, then use E = sqrt(E1^2+E2^2).", resultant, "N/C", template_id="perpendicular_fields")


def angled_fields(rng: random.Random) -> dict[str, str]:
    q1 = choose(rng, [1.0, 2.0, 3.5, 5.0, 7.0])
    q2 = choose(rng, [1.5, 2.5, 4.5, 6.5, 8.5])
    r_cm = choose(rng, [4, 5, 6, 8, 10, 12])
    angle = choose(rng, [45, 60, 90, 120])
    e1 = K_COULOMB * q1 * 1e-6 / (r_cm / 100) ** 2
    e2 = K_COULOMB * q2 * 1e-6 / (r_cm / 100) ** 2
    resultant = math.sqrt(e1**2 + e2**2 + 2 * e1 * e2 * math.cos(math.radians(angle)))
    q = (
        f"Two source charges q1 = {q1:g} uC and q2 = {q2:g} uC are both {r_cm:g} cm from point M. "
        f"The angle between their field directions at M is {angle:g} degrees. Calculate the resultant field."
    )
    return row("LD", q, "Use vector addition E = sqrt(E1^2+E2^2+2E1E2 cos theta).", resultant, "N/C", template_id="angled_fields")


def zero_field_between_charges(rng: random.Random) -> dict[str, str]:
    q1 = choose(rng, [1, 2, 3, 4, 6, 9])
    q2 = choose(rng, [2, 3, 5, 8, 12, 18])
    distance_cm = choose(rng, [12, 18, 24, 30, 40, 50, 60])
    x = distance_cm * math.sqrt(q1) / (math.sqrt(q1) + math.sqrt(q2))
    q = (
        f"Two positive charges q1 = {q1:g} uC and q2 = {q2:g} uC are separated by {distance_cm:g} cm. "
        "Find the distance from q1 to the point between them where the electric field is zero."
    )
    return row("LD", q, "Set kq1/x^2 = kq2/(d-x)^2 and solve for x.", x, "cm", template_id="zero_field_between_charges")


def charged_dust_equilibrium(rng: random.Random) -> dict[str, str]:
    mass_mg = choose(rng, [0.8, 1.2, 2.5, 4.0, 6.5, 9.0])
    charge_nc = choose(rng, [2, 3, 5, 8, 12, 20])
    field = mass_mg * 1e-6 * G / (charge_nc * 1e-9)
    q = f"A dust grain of mass {mass_mg:g} mg and charge {charge_nc:g} nC is held at rest by a vertical electric field. Find the required field magnitude."
    return row("LD", q, "For equilibrium qE = mg, so E = mg/q.", field, "N/C", template_id="charged_dust_equilibrium")


def electron_stopping_distance(rng: random.Random) -> dict[str, str]:
    speed = choose(rng, [1.5, 2.0, 2.8, 3.5, 4.2, 5.0, 6.0]) * 1e6
    field_kv = choose(rng, [1.5, 2.5, 4, 6, 8, 12])
    distance = E_MASS * speed**2 / (2 * E_CHARGE * field_kv * 1000)
    q = (
        f"An electron enters a uniform electric field of {field_kv:g} kV/m opposite to its motion with speed {speed / 1e6:g} x 10^6 m/s. "
        "Find the stopping distance."
    )
    return row("LD", q, "Use work-energy eEd = 1/2 mv^2.", distance, "m", template_id="electron_stopping_distance")


def lc_omega(rng: random.Random) -> dict[str, str]:
    l_mh = choose(rng, [5, 10, 20, 40, 75, 100, 150])
    c_uf = choose(rng, [0.5, 1, 2.2, 4.7, 10, 22, 47])
    omega = 1 / math.sqrt(l_mh * 1e-3 * c_uf * 1e-6)
    q = f"An ideal LC circuit has L = {l_mh:g} mH and C = {c_uf:g} uF. Calculate the angular frequency of free oscillation."
    return row("NL", q, "Use omega = 1/sqrt(LC).", omega, "rad/s", template_id="lc_omega")


def lc_frequency(rng: random.Random) -> dict[str, str]:
    l_mh = choose(rng, [2, 5, 8, 15, 30, 60, 120])
    c_uf = choose(rng, [0.22, 0.47, 1, 2.2, 4.7, 10])
    freq = 1 / (2 * math.pi * math.sqrt(l_mh * 1e-3 * c_uf * 1e-6))
    q = f"For an LC oscillator with inductance {l_mh:g} mH and capacitance {c_uf:g} uF, find the oscillation frequency."
    return row("NL", q, "Use f = 1/(2 pi sqrt(LC)).", freq, "Hz", template_id="lc_frequency")


def lc_current_from_energy(rng: random.Random) -> dict[str, str]:
    energy_mj = choose(rng, [0.4, 0.8, 1.5, 2.5, 4, 7, 12])
    l_mh = choose(rng, [5, 10, 20, 50, 80, 120])
    current = math.sqrt(2 * energy_mj * 1e-3 / (l_mh * 1e-3))
    q = f"The maximum magnetic energy in an LC circuit is {energy_mj:g} mJ and L = {l_mh:g} mH. Determine the maximum current."
    return row("NL", q, "At maximum current, W = 1/2 L Imax^2.", current, "A", template_id="lc_current_from_energy")


def lc_voltage_from_energy(rng: random.Random) -> dict[str, str]:
    energy_mj = choose(rng, [0.2, 0.5, 1, 2, 3.5, 6])
    c_uf = choose(rng, [0.5, 1, 2, 5, 10, 20, 50])
    voltage = math.sqrt(2 * energy_mj * 1e-3 / (c_uf * 1e-6))
    q = f"An LC circuit stores maximum electric energy {energy_mj:g} mJ in a capacitor of {c_uf:g} uF. Find the maximum capacitor voltage."
    return row("NL", q, "At maximum voltage, W = 1/2 C Umax^2.", voltage, "V", template_id="lc_voltage_from_energy")


def rlc_current(rng: random.Random) -> dict[str, str]:
    r = choose(rng, [10, 15, 20, 30, 47, 68, 100])
    l_mh = choose(rng, [20, 40, 60, 100, 150, 220])
    c_uf = choose(rng, [2.2, 4.7, 6.8, 10, 22, 47])
    f = choose(rng, [50, 60, 100, 200, 400, 800])
    u = choose(rng, [12, 24, 50, 110, 220])
    omega = 2 * math.pi * f
    xl = omega * l_mh * 1e-3
    xc = 1 / (omega * c_uf * 1e-6)
    z = math.sqrt(r**2 + (xl - xc) ** 2)
    current = u / z
    q = f"A series RLC circuit has R = {r:g} ohm, L = {l_mh:g} mH, C = {c_uf:g} uF and is driven by {u:g} V at {f:g} Hz. Calculate the RMS current."
    return row("CH", q, "Use Z = sqrt(R^2+(XL-XC)^2), I = U/Z.", current, "A", template_id="rlc_current")


def rlc_power_factor(rng: random.Random) -> dict[str, str]:
    r = choose(rng, [12, 18, 25, 40, 60, 90])
    l_mh = choose(rng, [30, 50, 80, 120, 200])
    c_uf = choose(rng, [3.3, 4.7, 10, 15, 33])
    f = choose(rng, [50, 100, 250, 500, 1000])
    omega = 2 * math.pi * f
    z = math.sqrt(r**2 + (omega * l_mh * 1e-3 - 1 / (omega * c_uf * 1e-6)) ** 2)
    cos_phi = r / z
    q = f"In a series RLC circuit R = {r:g} ohm, L = {l_mh:g} mH, C = {c_uf:g} uF at f = {f:g} Hz. Find the power factor."
    return row("CH", q, "Power factor in series RLC is cos phi = R/Z.", cos_phi, "-", template_id="rlc_power_factor")


def resonance_capacitance(rng: random.Random) -> dict[str, str]:
    l_mh = choose(rng, [10, 20, 50, 100, 200, 500])
    f = choose(rng, [50, 60, 100, 200, 400, 1000])
    capacitance_uf = 1 / ((2 * math.pi * f) ** 2 * l_mh * 1e-3) * 1e6
    q = f"What capacitance in uF is required for resonance at {f:g} Hz with an inductor L = {l_mh:g} mH?"
    return row("CH", q, "At resonance, C = 1/((2 pi f)^2 L).", capacitance_uf, "uF", template_id="resonance_capacitance")


def segment_rc_voltage(rng: random.Random) -> dict[str, str]:
    r = choose(rng, [20, 30, 50, 80, 120])
    c_uf = choose(rng, [5, 10, 20, 40, 80])
    f = choose(rng, [50, 100, 200, 500])
    current = choose(rng, [0.2, 0.35, 0.5, 0.8, 1.2])
    omega = 2 * math.pi * f
    xc = 1 / (omega * c_uf * 1e-6)
    voltage = current * math.sqrt(r**2 + xc**2)
    q = f"A current of {current:g} A flows through a series R-C section with R = {r:g} ohm, C = {c_uf:g} uF at {f:g} Hz. Find the RMS voltage across the section."
    return row("CH", q, "For an R-C section, U = I sqrt(R^2+XC^2).", voltage, "V", template_id="segment_rc_voltage")


def inductor_energy(rng: random.Random) -> dict[str, str]:
    l_mh = choose(rng, [5, 10, 20, 50, 100, 250, 500])
    current = choose(rng, [0.2, 0.5, 1.0, 1.5, 2.5, 4, 6])
    energy = 0.5 * l_mh * 1e-3 * current**2
    q = f"An inductor of {l_mh:g} mH carries current {current:g} A. Calculate the magnetic energy stored in it."
    return row("DDT", q, "Use W = 1/2 L I^2.", energy, "J", template_id="inductor_energy")


def self_induced_emf(rng: random.Random) -> dict[str, str]:
    l_mh = choose(rng, [20, 50, 80, 120, 200, 500])
    i1 = choose(rng, [0, 0.5, 1, 1.5, 2])
    delta_i = choose(rng, [0.8, 1.2, 2.0, 3.5, 5.0])
    dt_ms = choose(rng, [2, 5, 10, 20, 50])
    emf = l_mh * 1e-3 * delta_i / (dt_ms * 1e-3)
    q = f"Current in a {l_mh:g} mH coil changes from {i1:g} A to {i1 + delta_i:g} A in {dt_ms:g} ms. Find the magnitude of the self-induced emf."
    return row("DDT", q, "Use |e| = L |Delta I|/Delta t.", emf, "V", template_id="self_induced_emf")


def solenoid_b(rng: random.Random) -> dict[str, str]:
    turns = choose(rng, [300, 500, 800, 1000, 1500, 2000])
    length_cm = choose(rng, [20, 30, 40, 50, 80, 100])
    current = choose(rng, [0.5, 1, 1.5, 2, 3, 5])
    field = MU0 * (turns / (length_cm / 100)) * current
    q = f"A long solenoid has {turns:g} turns, length {length_cm:g} cm, and current {current:g} A. Calculate the magnetic field inside."
    return row("DDT", q, "Use B = mu0 (N/l) I.", field, "T", template_id="solenoid_b")


def solenoid_inductance(rng: random.Random) -> dict[str, str]:
    turns = choose(rng, [400, 600, 900, 1200, 1800])
    length_cm = choose(rng, [25, 40, 60, 80, 100])
    area_cm2 = choose(rng, [2, 4, 6, 10, 15, 25])
    inductance = MU0 * turns**2 * (area_cm2 * 1e-4) / (length_cm / 100)
    q = f"A solenoid has {turns:g} turns, length {length_cm:g} cm, and cross-sectional area {area_cm2:g} cm^2. Estimate its inductance in air."
    return row("DDT", q, "Use L = mu0 N^2 A/l.", inductance, "H", template_id="solenoid_inductance")


def magnetic_flux(rng: random.Random) -> dict[str, str]:
    field = choose(rng, [0.05, 0.08, 0.12, 0.2, 0.35, 0.5])
    area_cm2 = choose(rng, [10, 20, 50, 80, 120, 200])
    angle = choose(rng, [0, 30, 45, 60])
    flux = field * area_cm2 * 1e-4 * math.cos(math.radians(angle))
    q = f"A flat coil area is {area_cm2:g} cm^2 in a uniform magnetic field {field:g} T. The normal makes {angle:g} degrees with the field. Calculate the magnetic flux."
    return row("DDT", q, "Use Phi = B A cos(theta).", flux, "Wb", template_id="magnetic_flux")


def faraday_emf(rng: random.Random) -> dict[str, str]:
    turns = choose(rng, [20, 50, 100, 200, 500])
    phi1_mwb = choose(rng, [0.2, 0.5, 1.0, 1.5, 2.5])
    delta_mwb = choose(rng, [0.4, 0.8, 1.2, 2.0, 3.0])
    dt_ms = choose(rng, [5, 10, 20, 50, 100])
    emf = turns * delta_mwb * 1e-3 / (dt_ms * 1e-3)
    q = f"The flux through each turn of a {turns:g}-turn coil changes from {phi1_mwb:g} mWb to {phi1_mwb + delta_mwb:g} mWb in {dt_ms:g} ms. Find the average induced emf magnitude."
    return row("DDT", q, "Use |e| = N |Delta Phi|/Delta t.", emf, "V", template_id="faraday_emf")


def ohm_current(rng: random.Random) -> dict[str, str]:
    voltage = choose(rng, [3, 5, 9, 12, 24, 36, 48, 110, 220])
    resistance = choose(rng, [2, 4, 6, 8, 12, 15, 24, 40, 55, 100])
    current = voltage / resistance
    q = f"A resistor of {resistance:g} ohm is connected to a {voltage:g} V source. Calculate the current."
    return row("THCB", q, "Use Ohm law I = U/R.", current, "A", template_id="ohm_current")


def resistor_power(rng: random.Random) -> dict[str, str]:
    current = choose(rng, [0.2, 0.4, 0.75, 1.0, 1.5, 2.0, 3.0])
    resistance = choose(rng, [5, 10, 15, 22, 33, 47, 68])
    power = current**2 * resistance
    q = f"A current of {current:g} A flows through a {resistance:g} ohm resistor. Determine the electrical power dissipated."
    return row("THCB", q, "Use P = I^2 R.", power, "W", template_id="resistor_power")


def parallel_branch_currents(rng: random.Random) -> dict[str, str]:
    voltage = choose(rng, [6, 9, 12, 18, 24, 36, 48])
    r1 = choose(rng, [6, 8, 10, 12, 15, 20, 30])
    r2 = choose(rng, [10, 12, 18, 24, 30, 40, 60])
    i1 = voltage / r1
    i2 = voltage / r2
    q = f"Two resistors R1 = {r1:g} ohm and R2 = {r2:g} ohm are connected in parallel across {voltage:g} V. Find both branch currents I1 and I2."
    return row("THCB", q, "In parallel, each branch has the same voltage, so I1=U/R1 and I2=U/R2.", f"{fmt(i1)}; {fmt(i2)}", "A; A", template_id="parallel_branch_currents")


def series_voltage_divider(rng: random.Random) -> dict[str, str]:
    voltage = choose(rng, [9, 12, 18, 24, 36, 60])
    r1 = choose(rng, [5, 10, 15, 20, 30, 50])
    r2 = choose(rng, [10, 15, 25, 40, 60, 100])
    current = voltage / (r1 + r2)
    u1 = current * r1
    u2 = current * r2
    q = f"Resistors R1 = {r1:g} ohm and R2 = {r2:g} ohm are in series across {voltage:g} V. Calculate the voltage drops U1 and U2."
    return row("THCB", q, "Series current is I=U/(R1+R2), then U1=IR1 and U2=IR2.", f"{fmt(u1)}; {fmt(u2)}", "V; V", template_id="series_voltage_divider")


def parallel_equivalent(rng: random.Random) -> dict[str, str]:
    r1 = choose(rng, [4, 6, 8, 10, 12, 18, 24, 36])
    r2 = choose(rng, [6, 9, 12, 15, 20, 30, 40, 60])
    req = r1 * r2 / (r1 + r2)
    q = f"Calculate the equivalent resistance of two parallel resistors {r1:g} ohm and {r2:g} ohm."
    return row("THCB", q, "For two parallel resistors, Req=R1R2/(R1+R2).", req, "ohm", template_id="parallel_equivalent")


def relative_error(rng: random.Random) -> dict[str, str]:
    true_value = choose(rng, [10, 20, 25, 40, 50, 80, 100, 120])
    error = choose(rng, [-3, -2, -1.5, -1, 1, 1.5, 2, 3])
    measured = true_value + error
    percent = abs(error) / true_value * 100
    q = f"A true value is {true_value:g} cm and the measured value is {measured:g} cm. Find the relative error in percent."
    return row("THCB", q, "Relative error percent = |measured-true|/true * 100%.", percent, "%", template_id="relative_error")


def least_count_uncertainty(rng: random.Random) -> dict[str, str]:
    least_count = choose(rng, [0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    uncertainty = least_count / 2
    q = f"A length is measured with an instrument whose least count is {least_count:g} cm. Taking uncertainty as half the least count, find the absolute uncertainty."
    return row("THCB", q, "Absolute uncertainty = least count/2.", uncertainty, "cm", template_id="least_count_uncertainty")


TEMPLATES: list[Callable[[random.Random], dict[str, str]]] = [
    cap_energy,
    cap_charge,
    cap_voltage_from_charge,
    cap_series,
    cap_parallel,
    cap_disconnected_dielectric_voltage,
    cap_connected_dielectric_charge,
    cap_parallel_plate,
    cap_breakdown_charge,
    coulomb_force,
    point_field,
    force_in_uniform_field,
    perpendicular_fields,
    angled_fields,
    zero_field_between_charges,
    charged_dust_equilibrium,
    electron_stopping_distance,
    lc_omega,
    lc_frequency,
    lc_current_from_energy,
    lc_voltage_from_energy,
    rlc_current,
    rlc_power_factor,
    resonance_capacitance,
    segment_rc_voltage,
    inductor_energy,
    self_induced_emf,
    solenoid_b,
    solenoid_inductance,
    magnetic_flux,
    faraday_emf,
    ohm_current,
    resistor_power,
    parallel_branch_currents,
    series_voltage_divider,
    parallel_equivalent,
    relative_error,
    least_count_uncertainty,
]


def load_old_questions(paths: list[Path]) -> set[str]:
    questions: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            question_col = "question" if "question" in reader.fieldnames else None
            if question_col is None:
                question_col = next((name for name in reader.fieldnames if "question" in name.lower()), None)
            if question_col is None:
                continue
            for old_row in reader:
                questions.add(normalize_question(old_row.get(question_col, "")))
    return questions


def build_rows(count: int, seed: int, old_questions: set[str]) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    prefix_counts: dict[str, int] = {}
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 200:
            raise RuntimeError(f"Could only generate {len(rows)} unique rows after {attempts} attempts")
        template = TEMPLATES[len(rows) % len(TEMPLATES)]
        if rng.random() < 0.45:
            template = rng.choice(TEMPLATES)
        item = template(rng)
        normalized = normalize_question(item["question"])
        if normalized in old_questions or normalized in seen_questions:
            continue
        seen_questions.add(normalized)
        domain_prefix = item["id"]
        prefix_counts[domain_prefix] = prefix_counts.get(domain_prefix, 9000) + 1
        item["id"] = f"{domain_prefix}{prefix_counts[domain_prefix]}"
        rows.append(item)
    rng.shuffle(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic non-overlapping Type 2 physics test CSV.")
    parser.add_argument("--out", type=Path, default=Path("data/Synthetic_Type2_Test_1000.csv"))
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument(
        "--avoid",
        type=Path,
        nargs="*",
        default=[Path("data/Physics_Problems.csv"), Path("data/Test.csv"), Path("data/Physics_Questions_Only.csv")],
        help="Existing CSV files whose questions must not be reused exactly after normalization.",
    )
    args = parser.parse_args()

    old_questions = load_old_questions(args.avoid)
    rows = build_rows(args.count, args.seed, old_questions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "question", "cot", "answer", "unit"])
        writer.writeheader()
        writer.writerows(rows)

    prefix_counts: dict[str, int] = {}
    multi_answer = 0
    for item in rows:
        prefix = re.match(r"[A-Za-z]+", item["id"]).group(0)
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        if ";" in item["answer"]:
            multi_answer += 1

    print(f"wrote: {args.out}")
    print(f"rows: {len(rows)}")
    print(f"old_questions_avoided: {len(old_questions)}")
    print(f"multi_answer_rows: {multi_answer}")
    print("prefix_counts:")
    for prefix, value in sorted(prefix_counts.items()):
        print(f"  {prefix}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
