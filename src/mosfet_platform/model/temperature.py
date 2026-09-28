Boltzmann_constant = 1.380649e-23
elementary_charge = 1.602176634e-19

def thermal_voltage(temperature: float) -> float:
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    return Boltzmann_constant * temperature / elementary_charge