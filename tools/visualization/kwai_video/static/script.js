let items = [];
let currentIndex = 0;

document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const filePath = document.getElementById('filePath').value;
    const loadCount = document.getElementById('loadCount').value;
    
    if (!filePath) {
        alert('Please enter a file path');
        return;
    }

    try {
        const response = await fetch('/load_file', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                file_path: filePath,
                count: parseInt(loadCount)
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to load file');
        }

        const data = await response.json();
        items = data.items;
        currentIndex = 0;
        
        document.getElementById('contentSection').style.display = 'block';
        updateDisplay();
    } catch (error) {
        alert('Error loading file: ' + error.message);
    }
});

async function updateDisplay() {
    if (items.length === 0) return;

    const item = items[currentIndex];
    const mediaContainer = document.getElementById('mediaContainer');
    const responsesContainer = document.getElementById('responsesContainer');
    const itemCounter = document.getElementById('itemCounter');

    // Update counter
    itemCounter.textContent = `Item ${currentIndex + 1} of ${items.length}`;

    // Update media display
    try {
        const mediaInfo = await fetch(`/get_media_info/${item.__key__}`).then(res => res.json());
        
        if (mediaInfo.error) {
            mediaContainer.innerHTML = `<div class="alert alert-warning">${mediaInfo.error}</div>`;
            return;
        }

        if (mediaInfo.media_type === 'video') {
            mediaContainer.innerHTML = `
                <video controls>
                    <source src="/serve_media?path=${encodeURIComponent(mediaInfo.media_path)}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            `;
        } else {
            // Handle multiple images with carousel
            const paths = Array.isArray(mediaInfo.media_path) ? mediaInfo.media_path : [mediaInfo.media_path];
            
            if (paths.length === 1) {
                // Single image, no need for carousel
                mediaContainer.innerHTML = `<img src="/serve_media?path=${encodeURIComponent(paths[0])}" alt="Image" class="img-fluid">`;
            } else {
                // Multiple images, use carousel
                const carouselId = 'imageCarousel';
                const carouselItems = paths.map((path, index) => `
                    <div class="carousel-item ${index === 0 ? 'active' : ''}">
                        <img src="/serve_media?path=${encodeURIComponent(path)}" class="d-block w-100" alt="Image ${index + 1}">
                    </div>
                `).join('');

                mediaContainer.innerHTML = `
                    <div id="${carouselId}" class="carousel slide" data-bs-ride="carousel">
                        <div class="carousel-inner">
                            ${carouselItems}
                        </div>
                        <button class="carousel-control-prev" type="button" data-bs-target="#${carouselId}" data-bs-slide="prev">
                            <span class="carousel-control-prev-icon" aria-hidden="true"></span>
                            <span class="visually-hidden">Previous</span>
                        </button>
                        <button class="carousel-control-next" type="button" data-bs-target="#${carouselId}" data-bs-slide="next">
                            <span class="carousel-control-next-icon" aria-hidden="true"></span>
                            <span class="visually-hidden">Next</span>
                        </button>
                        <div class="carousel-indicators">
                            ${paths.map((_, index) => `
                                <button type="button" 
                                    data-bs-target="#${carouselId}" 
                                    data-bs-slide-to="${index}" 
                                    ${index === 0 ? 'class="active" aria-current="true"' : ''}
                                    aria-label="Slide ${index + 1}">
                                </button>
                            `).join('')}
                        </div>
                    </div>
                `;

                // Initialize the carousel
                new bootstrap.Carousel(document.getElementById(carouselId), {
                    interval: false, // 不自动轮播
                    touch: true,    // 允许触摸滑动
                    keyboard: true  // 允许键盘控制
                });
            }
        }
    } catch (error) {
        mediaContainer.innerHTML = `<div class="alert alert-danger">Error loading media: ${error.message}</div>`;
    }

    // Update responses
    responsesContainer.innerHTML = item.responses.map(response => 
        `<div class="response-item">${response}</div>`
    ).join('');
}

document.getElementById('prevBtn').addEventListener('click', () => {
    if (currentIndex > 0) {
        currentIndex--;
        updateDisplay();
    }
});

document.getElementById('nextBtn').addEventListener('click', () => {
    if (currentIndex < items.length - 1) {
        currentIndex++;
        updateDisplay();
    }
});

document.getElementById('loadMoreBtn').addEventListener('click', async () => {
    const filePath = document.getElementById('filePath').value;
    const loadCount = document.getElementById('loadCount').value;
    
    if (!filePath) {
        alert('Please enter a file path');
        return;
    }

    try {
        const response = await fetch('/load_file', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                file_path: filePath,
                start_index: items.length,  // Start from where we left off
                count: parseInt(loadCount)
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to load more items');
        }

        const data = await response.json();
        items = items.concat(data.items);  // Append new items to existing ones
        
        // Hide load more button if no more items
        if (!data.has_more) {
            document.getElementById('loadMoreBtn').style.display = 'none';
        }
        
        updateDisplay();
    } catch (error) {
        alert('Error loading more items: ' + error.message);
    }
}); 