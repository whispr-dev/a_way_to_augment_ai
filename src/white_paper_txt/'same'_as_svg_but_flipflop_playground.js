// JavaScript Implementation of a Universal Flip-Flop Playground UI
// Using HTML5 Canvas API for rendering the panel

const canvas = document.getElementById('flipFlopCanvas');
const ctx = canvas.getContext('2d');

// Draw Background
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, 800, 1000);
ctx.strokeStyle = 'black';
ctx.strokeRect(0, 0, 800, 1000);

// Draw Title
ctx.font = '24px Arial';
ctx.textAlign = 'center';
ctx.fillStyle = 'black';
ctx.fillText('Universal Flip-Flop Playground', 400, 40);

// NAND Gate Diagram Section
ctx.font = '18px Arial';
ctx.fillText('NAND Gate Diagram', 250, 120);

// First NAND Gate
ctx.strokeRect(60, 140, 120, 60);
ctx.fillText('NAND 1', 120, 175);
ctx.beginPath();
ctx.moveTo(60, 160);
ctx.lineTo(20, 160);
ctx.stroke();
ctx.fillText('Input X', 10, 165);
ctx.beginPath();
ctx.moveTo(60, 180);
ctx.lineTo(20, 180);
ctx.stroke();
ctx.fillText('Input Y', 10, 185);

// Second NAND Gate
ctx.strokeRect(260, 140, 120, 60);
ctx.fillText('NAND 2', 320, 175);
ctx.beginPath();
ctx.moveTo(380, 160);
ctx.lineTo(420, 160);
ctx.stroke();
ctx.fillText('Output O1', 430, 165);
ctx.beginPath();
ctx.moveTo(380, 180);
ctx.lineTo(420, 180);
ctx.stroke();
ctx.fillText('Output O2', 430, 185);

// Inputs & Outputs Section
ctx.fillText('Inputs & Outputs', 250, 300);

// Inputs
ctx.font = '12px Arial';
ctx.fillText('Inputs:', 80, 320);
ctx.beginPath();
ctx.arc(100, 350, 10, 0, 2 * Math.PI);
ctx.stroke();
ctx.fillText('Input X', 120, 355);

ctx.beginPath();
ctx.arc(100, 390, 10, 0, 2 * Math.PI);
ctx.stroke();
ctx.fillText('Input Y', 120, 395);

ctx.beginPath();
ctx.arc(100, 430, 10, 0, 2 * Math.PI);
ctx.stroke();
ctx.fillText('Input Z', 120, 435);

// Outputs
ctx.fillText('Outputs:', 350, 320);
ctx.beginPath();
ctx.arc(370, 350, 10, 0, 2 * Math.PI);
ctx.stroke();
ctx.fillText('Output Q', 390, 355);

ctx.beginPath();
ctx.arc(370, 390, 10, 0, 2 * Math.PI);
ctx.stroke();
ctx.fillText('Output Q̅', 390, 395);

// Configuration Section
ctx.font = '18px Arial';
ctx.fillText('Mode Configuration', 250, 500);
ctx.font = '12px Arial';
ctx.fillText('Mode Selector:', 80, 520);

ctx.strokeRect(160, 510, 40, 20);
ctx.fillText('SR', 180, 525);
ctx.strokeRect(210, 510, 40, 20);
ctx.fillText('JK', 230, 525);
ctx.strokeRect(260, 510, 40, 20);
ctx.fillText('D', 280, 525);
ctx.strokeRect(310, 510, 40, 20);
ctx.fillText('T', 330, 525);

// Status Indicators
ctx.font = '18px Arial';
ctx.fillText('State Indicators', 250, 700);
ctx.beginPath();
ctx.arc(150, 730, 10, 0, 2 * Math.PI);
ctx.fillStyle = 'red';
ctx.fill();
ctx.strokeStyle = 'black';
ctx.stroke();
ctx.fillStyle = 'black';
ctx.fillText('Q', 170, 735);

ctx.beginPath();
ctx.arc(250, 730, 10, 0, 2 * Math.PI);
ctx.fillStyle = 'green';
ctx.fill();
ctx.stroke();
ctx.fillStyle = 'black';
ctx.fillText('Q̅', 270, 735);
