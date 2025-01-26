import pkg_resources

# List installed packages
installed_packages = pkg_resources.working_set

# Filter packages with valid names and versions
clean_requirements = [
    f"{pkg.project_name}=={pkg.version}"
    for pkg in installed_packages
    if "@" not in pkg.location and "/croot/" not in pkg.location and "/work/" not in pkg.location
]

# Write clean requirements to a file
with open("clean_requirements.txt", "w") as f:
    f.write("\n".join(clean_requirements))

print("Clean requirements saved to clean_requirements.txt")
