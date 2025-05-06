const modeSelector = document.getElementById("mode-filter-selector");
const gpuConfigGroup = document.getElementById("gpu-config-group");
let filterSelected = null;
let processingMethod = null;

modeSelector.addEventListener("change", function () {
    const isGPU = modeSelector.value === "gpu";
    processingMethod = modeSelector.value;
    toggleGpuSettings(isGPU);
    saveFilterSelection();
});

function toggleGpuSettings(isGPU) {
    const gpuInputs = gpuConfigGroup.querySelectorAll("input");
    
    if (isGPU) {
        gpuConfigGroup.classList.add("active");
        gpuConfigGroup.style.opacity = "1";
        gpuInputs.forEach(input => input.disabled = false);
    } else {
        gpuConfigGroup.classList.remove("active");
        gpuConfigGroup.style.opacity = "0.4";
        gpuInputs.forEach(input => input.disabled = true);
    }
}

document.querySelectorAll('.filter-option').forEach(option => {
    option.addEventListener('click', () => {
        // Remove 'selected' from all
        document.querySelectorAll('.filter-option').forEach(el => el.classList.remove('selected'));

        // Add 'selected' to clicked one
        option.classList.add('selected');

        // Update hidden input
        const methodInput = document.getElementById('method-input');
        methodInput.value = option.getAttribute('data-value');

        // Update button text with selected filter name
        const filterName = option.querySelector('p')?.textContent || 'Procesar';
        document.getElementById('process-button').textContent = `Procesar ${filterName}`;

        // Save selection
        saveFilterSelection();
    });
});

function saveFilterSelection() {
    const selectedFilterOption = document.querySelector('.filter-option.selected');
    const selectedFilter = selectedFilterOption?.getAttribute('data-value') || 'mean';
    filterSelected = selectedFilter;

    console.log("Selected Filter:", selectedFilter);
}
