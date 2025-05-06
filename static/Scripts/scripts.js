const dropArea = document.getElementById("drop-area");
const inputFile = document.getElementById("image");
const imageView = document.getElementById("img-view");
let uploadedImageFile = null;
// Upload Image Function
inputFile.addEventListener("change", uploadImage);

function uploadImage() {
    let imgLink = URL.createObjectURL(inputFile.files[0]);
    imageView.style.backgroundImage = `url(${imgLink})`;
    imageView.style.backgroundSize = "cover";
    imageView.style.backgroundPosition = "center";
    imageView.textContent = "";  // Clear the default text
    imageView.style.border = 0;  // Remove border
    uploadedImageFile = inputFile.files[0];
}

// Select Filter Function

// Populate Mask Options
const MASK_OPTIONS = [
    { value: "9", label: "9×9", id: "mask9", default: true },
    { value: "13", label: "13×13", id: "mask13" },
    { value: "21", label: "21×21", id: "mask21" },
    { value: "custom", label: "Personalizada", id: "mask_custom" }
];

function populateMaskOptions() {
    const maskContainer = document.getElementById("mask-options");
    const customGroup = document.getElementById("custom-size-group");
    const customInput = customGroup.querySelector("input");

    // Clear existing options
    maskContainer.innerHTML = "";

    // Loop through each mask option and create a button for it
    MASK_OPTIONS.forEach(option => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "mask-option btn btn-outline-secondary";
        button.id = option.id;
        button.dataset.value = option.value;
        button.textContent = option.label;

        // Set the default active button if necessary
        if (option.default) {
            button.classList.add("active");
        }

        // Handle button click events
        button.addEventListener("click", () => {
            document.querySelectorAll(".mask-option").forEach(btn => btn.classList.remove("active"));
            button.classList.add("active");

            if (option.value === "custom") {
                customGroup.style.display = "flex";  // Show custom input
                customInput.disabled = false;  // Enable the input field
            } else {
                customGroup.style.display = "none";  // Hide custom input
                customInput.disabled = true;  // Disable the input field
            }
        });

        maskContainer.appendChild(button);
    });

    // Ensure correct initial state on load
    const activeDefault = MASK_OPTIONS.find(opt => opt.default);
    if (activeDefault?.value === "custom") {
        customGroup.style.display = "flex";
        customInput.disabled = false;
    } else {
        customGroup.style.display = "none";
        customInput.disabled = true;
    }
}

// Combine DOMContentLoaded into one event listener for better performance
document.addEventListener('DOMContentLoaded', () => {
    populateMaskOptions();
});
