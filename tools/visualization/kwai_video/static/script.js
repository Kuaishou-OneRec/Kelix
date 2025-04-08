let items = [];
let currentIndex = 0;
let currentCarousel = null;  // 保存当前的轮播图实例

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

        // 清除现有内容
        mediaContainer.innerHTML = '';

        // 处理媒体路径
        const getMediaUrl = (path) => `/serve_media?path=${encodeURIComponent(path)}`;

        if (mediaInfo.media_type === 'video') {
            const videoElement = document.createElement('video');
            videoElement.controls = true;
            videoElement.style.maxWidth = '100%';
            videoElement.style.maxHeight = '600px';
            
            const sourceElement = document.createElement('source');
            sourceElement.src = getMediaUrl(mediaInfo.media_path);
            sourceElement.type = 'video/mp4';
            
            videoElement.appendChild(sourceElement);
            mediaContainer.appendChild(videoElement);
        } else {
            // Handle multiple images with carousel
            const paths = Array.isArray(mediaInfo.media_path) ? mediaInfo.media_path : [mediaInfo.media_path];
            
            if (paths.length === 1) {
                const imgElement = document.createElement('img');
                imgElement.src = getMediaUrl(paths[0]);
                imgElement.className = 'img-fluid';
                imgElement.style.maxHeight = '600px';
                mediaContainer.appendChild(imgElement);
            } else {
                const carouselElement = document.createElement('div');
                carouselElement.id = 'imageCarousel';
                carouselElement.className = 'carousel slide';
                carouselElement.setAttribute('data-bs-ride', 'carousel');
                
                // 创建轮播指示器
                const indicators = document.createElement('div');
                indicators.className = 'carousel-indicators';
                paths.forEach((_, index) => {
                    const button = document.createElement('button');
                    button.type = 'button';
                    button.setAttribute('data-bs-target', '#imageCarousel');
                    button.setAttribute('data-bs-slide-to', index.toString());
                    if (index === 0) {
                        button.className = 'active';
                        button.setAttribute('aria-current', 'true');
                    }
                    button.setAttribute('aria-label', `Slide ${index + 1}`);
                    indicators.appendChild(button);
                });
                
                // 创建轮播内容
                const inner = document.createElement('div');
                inner.className = 'carousel-inner';
                paths.forEach((path, index) => {
                    const item = document.createElement('div');
                    item.className = `carousel-item ${index === 0 ? 'active' : ''}`;
                    
                    const img = document.createElement('img');
                    img.src = getMediaUrl(path);
                    img.className = 'd-block w-100';
                    img.alt = `Image ${index + 1}`;
                    img.style.maxHeight = '600px';
                    img.style.objectFit = 'contain';
                    
                    item.appendChild(img);
                    inner.appendChild(item);
                });
                
                // 创建控制按钮
                const prevButton = document.createElement('button');
                prevButton.className = 'carousel-control-prev';
                prevButton.type = 'button';
                prevButton.setAttribute('data-bs-target', '#imageCarousel');
                prevButton.setAttribute('data-bs-slide', 'prev');
                prevButton.innerHTML = `
                    <span class="carousel-control-prev-icon" aria-hidden="true"></span>
                    <span class="visually-hidden">Previous</span>
                `;
                
                const nextButton = document.createElement('button');
                nextButton.className = 'carousel-control-next';
                nextButton.type = 'button';
                nextButton.setAttribute('data-bs-target', '#imageCarousel');
                nextButton.setAttribute('data-bs-slide', 'next');
                nextButton.innerHTML = `
                    <span class="carousel-control-next-icon" aria-hidden="true"></span>
                    <span class="visually-hidden">Next</span>
                `;
                
                // 组装轮播图
                carouselElement.appendChild(indicators);
                carouselElement.appendChild(inner);
                carouselElement.appendChild(prevButton);
                carouselElement.appendChild(nextButton);
                
                mediaContainer.appendChild(carouselElement);
                
                // 初始化轮播图
                currentCarousel = new bootstrap.Carousel(carouselElement, {
                    interval: 3000
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