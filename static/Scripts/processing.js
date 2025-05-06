function processImage() {
    // Check if the image has already been processed
    if (document.querySelector("#output_image")) {
        alert("Presiona el boton Nueva Transformacion");
    } else {
        // Proceed with processing (your execute_filter function logic)
        execute_filter();
    }
}
function execute_filter() {
    if (!uploadedImageFile || !filterSelected) {
        alert("Please select both an image and a filter before processing.");
        return;
    }

    const maskSize = getMaskSize();
    if (maskSize === null) return;

    const formData = new FormData();
    formData.append("image", uploadedImageFile);
    formData.append("method", filterSelected);
    const modeSelect = document.getElementById("mode-filter-selector");
    const processingMethod = modeSelect.value; // Make sure this is up to date
    formData.append("mode", processingMethod);

    const maskOption = document.querySelector(".mask-option.active").dataset.value;
    formData.append("mask_option", maskOption);
    if (maskOption === "custom") {
        const customSize = document.querySelector('input[name="mask_size"]').value;
        formData.append("mask_size", customSize);
    }

    if (processingMethod === "gpu") {
        formData.append("blocks_x", document.querySelector('input[name="blocks_x"]').value);
        formData.append("blocks_y", document.querySelector('input[name="blocks_y"]').value);
        formData.append("threads_x", document.querySelector('input[name="threads_x"]').value);
        formData.append("threads_y", document.querySelector('input[name="threads_y"]').value);
    }

    fetch("/", {
        method: "POST",
        body: formData
    })
    .then(response => response.text())
    .then(html => {
        // Replace page content with response (processed result)
        document.open();
        document.write(html);
        document.close();
    })
    .catch(error => {
        alert("Error sending form: " + error.message);
        console.error(error);
    });
    populateMaskOptions() 
}


function getMaskSize() {
    const activeButton = document.querySelector(".mask-option.active");
    const selectedValue = activeButton?.dataset.value;

    if (!selectedValue) {
        alert("Por favor, selecciona un tamaño de máscara.");
        return null;
    }

    if (selectedValue === "custom") {
        const customInput = document.querySelector(".personalized-choice-text");
        const customValue = parseInt(customInput.value);

        if (isNaN(customValue) || customValue < 1 || customValue > 501 || customValue % 2 === 0) {
            alert("Por favor, ingresa un valor impar entre 1 y 501 para el tamaño personalizado.");
            return null;
        }

        return customValue;
    }

    return parseInt(selectedValue);
}
