// Enhanced script.js for x.ai-like experience

// Existing function with improved animation
function updateValue(id, value) {
    const valueSpan = document.getElementById(`${id}-value`);
    valueSpan.textContent = value;
    valueSpan.classList.add('highlight');
    setTimeout(() => valueSpan.classList.remove('highlight'), 200); // Faster highlight
}

// Handle page transitions
document.addEventListener('DOMContentLoaded', function() {
    // Add page content class to the content block
    const content = document.querySelector('.page-content');
    
    // Add event listeners to all forms for smooth submission
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            // Show loading spinner with smooth transition
            const spinner = document.querySelector('.loading-spinner');
            if (spinner) {
                spinner.style.display = 'flex';
                spinner.style.opacity = '1';
            }
            
            // If file upload is involved, show progress
            const fileInputs = form.querySelectorAll('input[type="file"]');
            if (fileInputs.length > 0) {
                const progressContainer = document.querySelector('.progress-container');
                if (progressContainer) {
                    progressContainer.style.display = 'block';
                    simulateProgress();
                }
            }
        });
    });

    // Enhanced animation for main title letters
    const letters = document.querySelectorAll('.main-title .letter');
    letters.forEach((letter, index) => {
        letter.addEventListener('mouseover', function() {
            this.style.transform = 'translateY(-10px)';
            this.style.color = 'rgba(255, 255, 255, 0.2)';
            this.style.webkitTextStroke = '1px #fff';
        });

        letter.addEventListener('mouseout', function() {
            this.style.transform = 'translateY(0)';
            this.style.color = 'transparent';
            this.style.webkitTextStroke = '1px rgba(255, 255, 255, 0.8)';
        });
    });

    // Parallax effect on scroll like x.ai
    window.addEventListener('scroll', function() {
        const scrollY = window.scrollY;
        
        // Move title with parallax effect
        const titleContainer = document.querySelector('.title-container');
        if (titleContainer) {
            titleContainer.style.transform = `translateY(${scrollY * 0.2}px)`;
        }
        
        // Move letters individually for more dynamic effect
        letters.forEach((letter, index) => {
            const factor = (index + 1) * 0.04;
            letter.style.transform = `translateY(${scrollY * factor}px)`;
        });
    });

    // x.ai style smooth navigation
    const navLinks = document.querySelectorAll('.site-header-main-nav a');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            // Only apply to internal links
            if (this.getAttribute('href').startsWith('/') || this.getAttribute('href').startsWith('./')) {
                e.preventDefault();
                const spinner = document.querySelector('.loading-spinner');
                if (spinner) {
                    spinner.style.display = 'flex';
                    spinner.style.opacity = '1';
                }
                
                // Fade out content
                const content = document.querySelector('.page-content');
                if (content) {
                    content.style.opacity = '0';
                    content.style.transition = 'opacity 0.2s ease'; // Faster transition
                }
                
                // Navigate after animation
                setTimeout(() => {
                    window.location = this.getAttribute('href');
                }, 200);
            }
        });
    });
    
    // Cursor follower effect like x.ai
    const body = document.querySelector('body');
    const cursor = document.createElement('div');
    cursor.classList.add('cursor-follower');
    body.appendChild(cursor);
    
    document.addEventListener('mousemove', function(e) {
        // Only show the follower on interactive elements
        const target = e.target;
        const isInteractive = 
            target.tagName === 'A' || 
            target.tagName === 'BUTTON' || 
            target.classList.contains('letter');
        
        if (isInteractive) {
            cursor.style.opacity = '1';
            // Add slight delay for smooth following
            requestAnimationFrame(() => {
                cursor.style.left = e.clientX + 'px';
                cursor.style.top = e.clientY + 'px';
            });
        } else {
            cursor.style.opacity = '0';
        }
    });

    // Create particles like x.ai
    const particlesContainer = document.getElementById('particles');
    if (particlesContainer) {
        for (let i = 0; i < 30; i++) {
            createParticle(particlesContainer);
        }
    }
});

// Simulate progress for file uploads
function simulateProgress() {
    const progressBar = document.querySelector('.progress-bar');
    if (!progressBar) return;
    
    let width = 0;
    const interval = setInterval(() => {
        if (width >= 100) {
            clearInterval(interval);
        } else {
            width += Math.random() * 15;
            if (width > 100) width = 100;
            progressBar.style.width = width + '%';
        }
    }, 200); // Faster progress
}

// Create floating particles like x.ai
function createParticle(container) {
    const particle = document.createElement('div');
    particle.classList.add('particle');
    
    // Random size
    const size = Math.random() * 4 + 1;
    particle.style.width = size + 'px';
    particle.style.height = size + 'px';
    
    // Random position
    const posX = Math.random() * 100;
    const posY = Math.random() * 100;
    particle.style.left = posX + '%';
    particle.style.top = posY + '%';
    
    // Random animation delay
    particle.style.animationDelay = Math.random() * 10 + 's';
    
    // Random animation duration
    const duration = Math.random() * 20 + 10;
    particle.style.animationDuration = duration + 's';
    
    // Add to container
    container.appendChild(particle);
}

// Function to initialize scrollable tables
function initScrollableTables() {
    const tables = document.querySelectorAll('.table-container table');
    tables.forEach(table => {
        // Check if table is larger than container
        if (table.offsetWidth > table.parentElement.offsetWidth || table.offsetHeight > 400) {
            table.parentElement.style.overflowX = 'auto';
            table.parentElement.style.overflowY = 'auto';
        }
    });
}

// Add CSS to handle the cursor follower
document.addEventListener('DOMContentLoaded', function() {
    const style = document.createElement('style');
    style.textContent = `
        .cursor-follower {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background-color: rgba(255, 255, 255, 0.1);
            position: fixed;
            pointer-events: none;
            z-index: 9999;
            transform: translate(-50%, -50%);
            transition: opacity 0.2s ease; /* Faster transition */
            opacity: 0;
            mix-blend-mode: difference;
        }
    `;
    document.head.appendChild(style);
});

// Handle lazy loading for batch predictions
window.addEventListener('load', function() {
    // Initialize scrollable tables
    initScrollableTables();
    
    // Hide spinner when page is fully loaded
    const spinner = document.querySelector('.loading-spinner');
    if (spinner) {
        spinner.style.opacity = '0';
        setTimeout(() => spinner.style.display = 'none', 300);
    }
});