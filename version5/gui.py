#!/usr/bin/env python3
"""
Roshn Demarcation Plan Validation System - Professional GUI Application
=========================================================================

A modern, themed graphical interface for validating ROSHN demarcation plan documents.

Features:
- ROSHN brand colors and professional theme
- PDF file/folder browsing
- Excel ground truth file selection
- AI analysis toggle with cost estimation
- Real-time progress monitoring
- Validation results summary
- Excel report generation

Author: ROSHN Development Team
Version: 5.0
"""

import os
import sys
import threading
import queue
import logging
import base64
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# EMBEDDED ROSHN LOGO (SVG as string)
# ============================================================================
ROSHN_LOGO_SVG = '''<?xml version="1.0" encoding="UTF-8"?>
<svg id="Layer_1" xmlns="http://www.w3.org/2000/svg" version="1.1" viewBox="0 0 999.21 293.04">
  <!-- Generator: Adobe Illustrator 29.0.1, SVG Export Plug-In . SVG Version: 2.1.0 Build 192)  -->
  <defs>
    <style>
      .st0 {
        fill: #006450;
      }
    </style>
  </defs>
  <g>
    <g>
      <g>
        <g>
          <path class="st0" d="M28.5,185.53c0-1.68-.44-3.25-1.16-4.66l22.93-22.93c2.03,1.23,4.38,1.98,6.93,1.98,7.4,0,13.41-6,13.41-13.41,0-7.4-6-13.41-13.41-13.41-2.54,0-4.9.75-6.93,1.98l-22.93-22.93c.72-1.4,1.16-2.97,1.16-4.65,0-5.69-4.62-10.31-10.31-10.31-5.69,0-10.31,4.62-10.31,10.31,0,5.69,4.62,10.31,10.31,10.31,1.69,0,3.25-.44,4.66-1.16l22.93,22.93c-1.23,2.03-1.98,4.38-1.98,6.93s.75,4.9,1.98,6.93l-22.93,22.93c-1.4-.72-2.97-1.16-4.66-1.16-5.69,0-10.31,4.62-10.31,10.31,0,5.69,4.62,10.31,10.31,10.31,5.69,0,10.31-4.62,10.31-10.31Z"/>
          <polygon class="st0" points="0 146.52 15.06 161.59 15.06 131.45 0 146.52"/>
        </g>
        <g>
          <path class="st0" d="M264.54,185.53c0-1.68.44-3.25,1.16-4.66l-22.93-22.93c-2.03,1.23-4.38,1.98-6.93,1.98-7.4,0-13.41-6-13.41-13.41s6-13.41,13.41-13.41c2.54,0,4.9.75,6.93,1.98l22.93-22.93c-.72-1.4-1.16-2.97-1.16-4.65,0-5.69,4.62-10.31,10.31-10.31s10.31,4.62,10.31,10.31-4.62,10.31-10.31,10.31c-1.69,0-3.25-.44-4.66-1.16l-22.93,22.93c1.23,2.03,1.98,4.38,1.98,6.93s-.75,4.9-1.98,6.93l22.93,22.93c1.4-.72,2.97-1.16,4.66-1.16,5.69,0,10.31,4.62,10.31,10.31s-4.62,10.31-10.31,10.31-10.31-4.62-10.31-10.31Z"/>
          <polygon class="st0" points="293.04 146.52 277.98 161.59 277.98 131.45 293.04 146.52"/>
        </g>
      </g>
      <g>
        <g>
          <path class="st0" d="M90.65,257.55c-1.19-1.19-2.61-1.99-4.11-2.47v-32.43c2.31-.56,4.5-1.7,6.3-3.5,5.23-5.23,5.23-13.73,0-18.96-5.23-5.23-13.73-5.23-18.96,0-1.8,1.8-2.93,3.99-3.5,6.3h-32.43c-.49-1.5-1.28-2.92-2.47-4.11-4.03-4.03-10.56-4.03-14.58,0-4.03,4.03-4.03,10.56,0,14.58,4.03,4.03,10.56,4.03,14.58,0,1.19-1.19,1.99-2.61,2.47-4.11h32.43c.56,2.3,1.7,4.5,3.5,6.3,1.8,1.8,3.99,2.93,6.3,3.5v32.43c-1.5.49-2.92,1.28-4.11,2.47-4.03,4.03-4.03,10.56,0,14.58,4.03,4.03,10.56,4.03,14.58,0,4.03-4.03,4.03-10.56,0-14.58Z"/>
          <polygon class="st0" points="42.91 250.12 64.22 250.13 42.91 228.82 42.91 250.12"/>
        </g>
        <g>
          <path class="st0" d="M257.55,90.65c-1.19-1.19-1.99-2.61-2.47-4.11h-32.43c-.56,2.31-1.7,4.5-3.5,6.3-5.23,5.23-13.73,5.23-18.96,0-5.23-5.23-5.23-13.73,0-18.96,1.8-1.8,3.99-2.93,6.3-3.5v-32.43c-1.5-.49-2.92-1.28-4.11-2.47-4.03-4.03-4.03-10.56,0-14.58,4.03-4.03,10.56-4.03,14.58,0,4.03,4.03,4.03,10.56,0,14.58-1.19,1.19-2.61,1.99-4.11,2.47v32.43c2.3.56,4.5,1.7,6.3,3.5,1.8,1.8,2.93,3.99,3.5,6.3h32.43c.49-1.5,1.28-2.92,2.47-4.11,4.03-4.03,10.56-4.03,14.58,0,4.03,4.03,4.03,10.56,0,14.58-4.03,4.03-10.56,4.03-14.58,0Z"/>
          <polygon class="st0" points="250.12 42.91 250.13 64.22 228.82 42.91 250.12 42.91"/>
        </g>
      </g>
      <g>
        <g>
          <path class="st0" d="M35.48,90.65c1.19-1.19,1.99-2.61,2.47-4.11h32.43c.56,2.31,1.7,4.5,3.5,6.3,5.23,5.23,13.73,5.23,18.96,0,5.23-5.23,5.23-13.73,0-18.96-1.8-1.8-3.99-2.93-6.3-3.5v-32.43c1.5-.49,2.92-1.28,4.11-2.47,4.03-4.03,4.03-10.56,0-14.58-4.03-4.03-10.56-4.03-14.58,0-4.03,4.03-4.03,10.56,0,14.58,1.19,1.19,2.61,1.99,4.11,2.47v32.43c-2.3.56-4.5,1.7-6.3,3.5-1.8,1.8-2.93,3.99-3.5,6.3h-32.43c-.49-1.5-1.28-2.92-2.47-4.11-4.03-4.03-10.56-4.03-14.58,0-4.03,4.03-4.03,10.56,0,14.58,4.03,4.03,10.56,4.03,14.58,0Z"/>
          <polygon class="st0" points="42.91 42.91 42.91 64.22 64.22 42.91 42.91 42.91"/>
        </g>
        <g>
          <path class="st0" d="M202.39,257.55c1.19-1.19,2.61-1.99,4.11-2.47v-32.43c-2.31-.56-4.5-1.7-6.3-3.5-5.23-5.23-5.23-13.73,0-18.96,5.23-5.23,13.73-5.23,18.96,0,1.8,1.8,2.93,3.99,3.5,6.3h32.43c.49-1.5,1.28-2.92,2.47-4.11,4.03-4.03,10.56-4.03,14.58,0,4.03,4.03,4.03,10.56,0,14.58-4.03,4.03-10.56,4.03-14.58,0-1.19-1.19-1.99-2.61-2.47-4.11h-32.43c-.56,2.3-1.7,4.5-3.5,6.3-1.8,1.8-3.99,2.93-6.3,3.5v32.43c1.5.49,2.92,1.28,4.11,2.47,4.03,4.03,4.03,10.56,0,14.58-4.03,4.03-10.56,4.03-14.58,0-4.03-4.03-4.03-10.56,0-14.58Z"/>
          <polygon class="st0" points="250.12 250.12 228.82 250.13 250.13 228.82 250.12 250.12"/>
        </g>
      </g>
      <g>
        <g>
          <path class="st0" d="M185.53,264.54c-1.68,0-3.25.44-4.66,1.16l-22.93-22.93c1.23-2.03,1.98-4.38,1.98-6.93,0-7.4-6-13.41-13.41-13.41-7.4,0-13.41,6-13.41,13.41,0,2.54.75,4.9,1.98,6.93l-22.93,22.93c-1.4-.72-2.97-1.16-4.65-1.16-5.69,0-10.31,4.62-10.31,10.31,0,5.69,4.62,10.31,10.31,10.31,5.69,0,10.31-4.62,10.31-10.31,0-1.69-.44-3.25-1.16-4.66l22.93-22.93c2.03,1.23,4.38,1.98,6.93,1.98,2.54,0,4.9-.75,6.93-1.98l22.93,22.93c-.72,1.4-1.16,2.97-1.16,4.66,0,5.69,4.62,10.31,10.31,10.31,5.69,0,10.31-4.62,10.31-10.31,0-5.69-4.62-10.31-10.31-10.31Z"/>
          <polygon class="st0" points="146.52 293.04 161.59 277.98 131.45 277.98 146.52 293.04"/>
        </g>
        <g>
          <path class="st0" d="M185.53,28.5c-1.68,0-3.25-.44-4.66-1.16l-22.93,22.93c1.23,2.03,1.98,4.38,1.98,6.93,0,7.4-6,13.41-13.41,13.41-7.4,0-13.41-6-13.41-13.41,0-2.54.75-4.9,1.98-6.93l-22.93-22.93c-1.4.72-2.97,1.16-4.65,1.16-5.69,0-10.31-4.62-10.31-10.31,0-5.69,4.62-10.31,10.31-10.31,5.69,0,10.31,4.62,10.31,10.31,0,1.69-.44,3.25-1.16,4.66l22.93,22.93c2.03-1.23,4.38-1.98,6.93-1.98,2.54,0,4.9.75,6.93,1.98l22.93-22.93c-.72-1.4-1.16-2.97-1.16-4.66,0-5.69,4.62-10.31,10.31-10.31,5.69,0,10.31,4.62,10.31,10.31,0,5.69-4.62,10.31-10.31,10.31Z"/>
          <polygon class="st0" points="146.52 0 161.59 15.06 131.45 15.06 146.52 0"/>
        </g>
      </g>
    </g>
    <g>
      <path class="st0" d="M142.7,136.41l3.82,9.78,3.81-9.77c.89-2.26,1.64-4.55,2.24-6.79,3.56-13.3,2.78-27.05-2.25-39.76l-3.8-9.62-3.81,9.62c-2.95,7.45-4.45,15.25-4.45,23.19,0,5.55.74,11.12,2.2,16.57.6,2.25,1.36,4.53,2.24,6.78Z"/>
      <path class="st0" d="M136.9,141.97l9.62,4.21-4.22-9.61c-.97-2.23-2.05-4.38-3.21-6.39-6.89-11.93-17.16-21.1-29.7-26.52l-9.49-4.11,4.1,9.5c3.18,7.36,7.64,13.93,13.25,19.54,3.92,3.92,8.39,7.34,13.27,10.16,2.01,1.16,4.16,2.24,6.38,3.21Z"/>
      <path class="st0" d="M136.74,150.01l9.78-3.82-9.77-3.81c-2.26-.89-4.55-1.64-6.79-2.24-13.3-3.56-27.05-2.78-39.76,2.25l-9.62,3.8,9.62,3.81c7.45,2.95,15.25,4.45,23.19,4.45,5.55,0,11.12-.74,16.57-2.2,2.25-.6,4.53-1.36,6.78-2.24Z"/>
      <path class="st0" d="M142.3,155.8l4.21-9.62-9.61,4.22c-2.23.97-4.38,2.05-6.39,3.21-11.93,6.89-21.1,17.16-26.52,29.7l-4.11,9.49,9.5-4.1c7.36-3.18,13.93-7.64,19.54-13.25,3.92-3.92,7.34-8.39,10.16-13.27,1.16-2.01,2.24-4.16,3.21-6.38Z"/>
      <path class="st0" d="M150.34,155.97l-3.82-9.78-3.81,9.77c-.89,2.26-1.64,4.55-2.24,6.79-3.56,13.3-2.78,27.05,2.25,39.76l3.8,9.62,3.81-9.62c2.95-7.45,4.45-15.25,4.45-23.19,0-5.55-.74-11.12-2.2-16.57-.6-2.25-1.36-4.53-2.24-6.78Z"/>
      <path class="st0" d="M156.13,150.4l-9.62-4.21,4.22,9.61c.97,2.23,2.05,4.38,3.21,6.39,6.89,11.93,17.16,21.1,29.7,26.52l9.49,4.11-4.1-9.5c-3.18-7.36-7.64-13.93-13.25-19.54-3.92-3.92-8.39-7.34-13.27-10.16-2.01-1.16-4.16-2.24-6.38-3.21Z"/>
      <path class="st0" d="M156.3,142.37l-9.78,3.82,9.77,3.81c2.26.89,4.55,1.64,6.79,2.24,13.3,3.56,27.05,2.78,39.76-2.25l9.62-3.8-9.62-3.81c-7.45-2.95-15.25-4.45-23.19-4.45-5.55,0-11.12.74-16.57,2.2-2.25.6-4.53,1.36-6.78,2.24Z"/>
      <path class="st0" d="M150.73,136.57l-4.21,9.62,9.61-4.22c2.23-.97,4.38-2.05,6.39-3.21,11.93-6.89,21.1-17.16,26.52-29.7l4.11-9.49-9.5,4.1c-7.36,3.18-13.93,7.64-19.54,13.25-3.92,3.92-7.34,8.39-10.16,13.27-1.16,2.01-2.24,4.16-3.21,6.38Z"/>
    </g>
  </g>
  <g>
    <g>
      <path class="st0" d="M536.22,206.55h-.01c-2.6-1.82-5.49-3.58-8.62-5.24-3.19-1.7-5.9-3.25-8.3-4.77-2.46-1.53-4.53-3.15-6.15-4.78-1.38-1.4-2.05-2.88-2.05-4.53,0-2.73.69-4.69,2.03-5.86,2.96-2.59,7.78-2.5,13.2.22,2.8,1.42,5.44,3.48,7.85,6.13l1.67,1.89,5.18-9.41-1.43-.89c-2.52-1.58-5.3-2.75-8.24-3.49-5.35-1.43-11.16-1.5-16.34-.24-2.52.62-4.76,1.58-6.63,2.82-1.94,1.26-3.53,2.89-4.74,4.83-1.21,2.01-1.82,4.31-1.82,6.82,0,3.28.99,6.15,2.98,8.53,1.81,2.21,4.08,4.25,6.72,6.07,2.54,1.73,5.38,3.43,8.44,5.02,2.98,1.57,5.7,3.17,8.1,4.79,2.37,1.59,4.39,3.4,5.98,5.38,1.45,1.77,2.17,3.88,2.17,6.4,0,1.61-.29,2.97-.89,4.17-.6,1.2-1.43,2.23-2.44,3.05-1.01.83-2.26,1.48-3.72,1.95-4.37,1.38-9.43.73-14.57-1.94-3.34-1.73-6.25-4.54-8.63-8.36l-1.53-2.42-5.99,9.87-.41.65,1.44.93c3.24,2.09,6.9,3.72,10.87,4.84,4.04,1.13,8.11,1.7,12.07,1.7,3,0,5.94-.34,8.73-1.03,2.84-.71,5.38-1.86,7.53-3.4,2.21-1.58,3.99-3.61,5.32-6.03,1.33-2.46,2-5.41,2-8.76,0-3.17-.99-6.04-2.96-8.55-1.78-2.27-4.08-4.4-6.81-6.34Z"/>
      <path class="st0" d="M605.07,184.9c.13-2.21.37-4.09.73-5.58.23-1.01.62-1.55.62-1.55l2.71-3.08-19.38.03,2.51,3h.01c.29.42.53.97.66,1.58.34,1.5.58,3.33.72,5.44.15,2.15.23,4.6.27,7.29.03,2.6.04,5,.04,7.14v5.81h-26.92v-5.45c0-2.14.01-4.51.05-7.19.03-2.8.12-5.23.25-7.43.13-2.21.37-4.09.73-5.58.23-1.01.64-1.55.64-1.55l2.68-3.08-19.42.03,2.57,3.05.04.05c.09.12.42.62.62,1.47.34,1.5.58,3.33.72,5.44.15,2.15.23,4.6.27,7.29.03,2.6.04,5,.04,7.14v17.08c0,2.14-.01,4.52-.04,7.14-.04,2.69-.12,5.15-.27,7.29-.13,2.14-.37,3.97-.72,5.44-.23.98-.64,1.5-.64,1.5h.01l-2.81,3.09,19.5-.03-2.52-2.98h0c-.31-.42-.54-.99-.68-1.62-.36-1.47-.6-3.36-.73-5.57-.13-2.2-.23-4.63-.25-7.43-.04-2.67-.05-5.04-.05-7.18v-5.16h26.92v5.5c0,2.14-.01,4.52-.04,7.14-.04,2.69-.12,5.15-.27,7.29-.13,2.14-.37,3.97-.72,5.44-.23.98-.62,1.51-.62,1.51l-2.76,3.06h19.44s-2.53-3-2.53-3c-.29-.42-.53-.98-.66-1.62-.36-1.47-.6-3.36-.73-5.57-.13-2.15-.21-4.64-.21-7.4v-30.8c0-2.77.08-5.2.21-7.4Z"/>
      <path class="st0" d="M659.32,177.71c.09.15.4.68.61,1.6.33,1.53.57,3.46.7,5.58.16,2.19.24,4.62.28,7.43.03,2.65.04,5.05.04,7.19v21.25l-33.01-45.73-.25-.33h-13.44s2.45,3.04,2.45,3.04l.05.05c.08.13.42.65.62,1.54.34,1.51.58,3.38.72,5.57.15,2.23.23,4.72.23,7.4v30.8c0,2.72-.08,5.15-.23,7.4-.13,2.24-.37,4.11-.72,5.57-.24,1.02-.65,1.58-.64,1.58l-2.65,3.05,12.98-.03-2.18-2.97c-.09-.13-.4-.68-.62-1.62-.33-1.49-.57-3.36-.7-5.58-.16-2.19-.24-4.62-.28-7.43-.03-2.64-.04-5.04-.04-7.18v-27.76l37.99,52.54h4.64s0-41.16,0-41.16c0-2.14.01-4.54.04-7.19.04-2.76.12-5.19.27-7.43.13-2.21.37-4.09.73-5.58.11-.45.29-1.11.62-1.66l2.25-2.97-12.65.03,2.18,3Z"/>
      <path class="st0" d="M424.73,237.15c-.5-.23-2.29-1.06-3.85-2.04-1.98-1.27-3.94-2.73-5.82-4.32-2-1.68-4.03-3.57-5.99-5.57-1.9-1.9-3.69-3.89-5.33-5.89-1.54-1.88-2.77-3.59-3.75-5.21-1.11-1.84-1.25-2.67-1.25-2.94,0-.38.24-.78.68-1.19.81-.68,1.86-1.38,3.1-2.1,1.37-.77,2.84-1.63,4.38-2.56,1.6-.94,3.17-2.1,4.67-3.41,1.5-1.33,2.76-2.92,3.77-4.71,1.05-1.9,1.57-4.11,1.57-6.59,0-2.07-.46-4.11-1.38-6.09-.9-1.94-2.37-3.67-4.34-5.17-1.88-1.45-4.3-2.6-7.19-3.42-2.85-.82-6.26-1.23-10.13-1.23h-27.32s2.51,3.01,2.51,3.01h.01c.29.42.53.99.68,1.62.34,1.51.58,3.38.72,5.57.13,2.2.21,4.63.21,7.4l.09,23.94c0,2.14-.01,4.52-.04,7.14-.04,2.69-.13,5.08-.27,7.29-.13,2.2-.37,4.03-.72,5.44-.23.98-.64,1.5-.64,1.5h.01l-2.81,3.09,19.5-.03-2.52-2.98h-.01c-.29-.42-.53-.99-.68-1.6-.34-1.54-.58-3.41-.72-5.58-.15-2.24-.23-4.67-.27-7.43-.03-2.64-.04-5.04-.04-7.18v-36.7h12.3c3.93,0,6.78,1.09,8.71,3.3,1.96,2.24,2.97,5.38,2.97,9.36,0,2.88-.6,5.25-1.76,7.03-1.34,2-2.8,3.73-4.35,5.13l-4.67,4.27c-1.83,1.66-2.75,3.61-2.75,5.81,0,1.78.48,3.77,1.54,6.25.94,2.19,2.21,4.51,3.81,6.88,1.58,2.33,3.4,4.66,5.42,6.92,2.1,2.36,4.3,4.43,6.53,6.15l.49.37h22.93l-7.97-3.51h-.03Z"/>
      <path class="st0" d="M482.45,182.95c-2.98-2.79-6.47-4.88-10.32-6.24-7.68-2.7-15.78-2.7-23.43,0-3.85,1.34-7.33,3.44-10.34,6.24-2.98,2.77-5.42,6.25-7.26,10.33-1.84,4.08-2.75,8.93-2.75,14.41s.92,10.27,2.75,14.37c1.82,4.09,4.27,7.58,7.27,10.34,2.97,2.73,6.44,4.82,10.3,6.22,3.86,1.38,7.82,2.07,11.73,2.07s7.87-.69,11.73-2.07c3.88-1.4,7.33-3.49,10.3-6.22,3-2.77,5.45-6.25,7.26-10.34,1.85-4.14,2.78-8.98,2.78-14.37s-.93-10.29-2.78-14.41c-1.84-4.09-4.29-7.58-7.26-10.33ZM480.76,218.04c-.77,4-2.07,7.36-3.85,10-1.77,2.61-3.94,4.63-6.47,5.98-4.73,2.51-9.72,2.69-14.82.56-2.27-.94-4.44-2.42-6.47-4.41-2.02-1.98-3.81-4.57-5.34-7.69-1.53-3.12-2.7-6.9-3.48-11.25-.93-5.22-1.02-9.9-.27-13.91.77-3.98,2.05-7.35,3.85-10,1.76-2.61,3.94-4.64,6.47-6.01,2.51-1.36,5.22-2.05,7.96-2.05,2.3,0,4.6.48,6.85,1.44,2.3.96,4.44,2.45,6.39,4.44,2,2.02,3.81,4.63,5.38,7.75,1.58,3.19,2.77,6.96,3.51,11.25.93,5.2,1.02,9.88.29,13.9Z"/>
      <path class="st0" d="M728.22,211.1h6.57c1.52,0,2.65.15,3.34.43.63.24,1.1.58,1.38.96.29.38.46.87.51,1.5.08.88.12,1.9.12,3.05v6.34c0,1.98-.37,3.71-1.14,5.27-.81,1.6-1.85,2.93-3.14,3.95-1.38,1.12-2.91,1.96-4.72,2.57-1.83.63-3.74.95-5.69.95-3.27,0-6.31-.79-9.02-2.35-2.77-1.58-5.24-3.73-7.34-6.38-2.1-2.66-3.82-5.75-5.13-9.18-1.32-3.46-2.2-7.03-2.59-10.58-.41-3.53-.32-7.07.29-10.52.55-3.28,1.7-6.31,3.39-9,1.65-2.59,3.87-4.68,6.61-6.21,2.74-1.56,6.22-2.35,10.37-2.35,3.14,0,6.34.77,9.52,2.27,3.21,1.51,6.16,3.76,8.79,6.67l1.57,1.7,5.61-8.77-1.64-.91c-3.32-1.9-6.86-3.36-10.53-4.33-7.99-2.13-16.27-1.94-23.86.54-4.14,1.35-7.83,3.44-10.99,6.17-3.17,2.74-5.75,6.24-7.67,10.38-1.89,4.15-2.85,9.04-2.85,14.52,0,5.04.91,9.64,2.72,13.66,1.78,4,4.23,7.5,7.3,10.39,3.03,2.85,6.59,5.07,10.6,6.58,3.91,1.5,8.13,2.25,12.54,2.25s9.11-.84,13.62-2.53c4.54-1.7,8.92-4.4,13.04-8.01l.59-.52v-.81c0-4.18.03-7.82.09-11,.05-3.08.33-6.08.84-8.95l.33-2.07h-23.43v4.32Z"/>
      <path class="st0" d="M997.21,185.29c-1.67-3.1-4.5-5.68-8.4-7.69-3.85-1.94-8.89-2.9-15-2.9h-26.65s2.48,3.04,2.48,3.04l.03.04c.07.11.41.62.62,1.55.34,1.51.58,3.38.72,5.57.15,2.23.23,4.72.23,7.4v30.8c0,2.72-.08,5.15-.23,7.4-.13,2.24-.37,4.11-.72,5.57-.24,1.01-.64,1.58-.62,1.58l-2.67,3.04h19.42s-2.63-3.04-2.63-3.04c0-.01-.42-.53-.64-1.54-.32-1.33-.54-3.06-.72-5.44-.15-2.15-.23-4.6-.27-7.29v-.73c-.04-2.32-.05-4.47-.05-6.41v-37.04h11.78c3.29,0,6.06.73,8.24,2.14,2.23,1.47,3.81,3.26,4.81,5.48,1.05,2.27,1.5,4.75,1.39,7.6-.09,2.83-.85,5.65-2.27,8.42-1.38,2.75-3.46,5.32-6.22,7.64-2.75,2.36-8.73,6.13-8.79,6.17l-7.77,4.91,9.07-1.51c5.61-.95,8.69-2.68,12.75-5.21,4.03-2.49,7.27-5.42,9.64-8.71,2.41-3.3,3.86-6.84,4.3-10.5.46-3.7-.15-7.18-1.84-10.31Z"/>
      <path class="st0" d="M932.96,177.65h.01c.29.42.52.97.66,1.57.34,1.56.58,3.44.7,5.47.13,2.22.22,4.6.26,7.28.03,2.6.04,4.95.04,7.04v19.18c0,2.65-.52,5.04-1.52,7.1-1.01,2.05-2.39,3.8-4.07,5.15-1.74,1.43-3.68,2.51-5.8,3.21-4.16,1.47-8.67,1.53-12.71.14-1.91-.64-3.64-1.65-5.16-2.98-1.51-1.38-2.72-3.11-3.6-5.16-.92-2.11-1.38-4.55-1.38-7.46v-19.54c0-2.09.01-4.45.04-6.99.04-2.64.12-5.04.26-7.15.14-2.14.38-3.94.7-5.32.23-1.05.62-1.6.62-1.6l2.25-2.92-18.3.03,2.16,2.94c.08.13.38.65.58,1.59.34,1.48.57,3.31.7,5.46.14,2.2.22,4.58.26,7.28.03,2.6.04,4.95.04,7.04v19.18c0,3.68.73,6.99,2.16,9.85,1.44,2.83,3.35,5.2,5.69,7.04,2.33,1.86,5,3.28,7.94,4.19,2.95.92,5.99,1.39,9.05,1.39s6.08-.47,9.03-1.39c2.94-.91,5.63-2.31,7.99-4.19,2.38-1.86,4.32-4.24,5.74-7.07,1.43-2.87,2.13-6.19,2.05-9.84v-19.17c0-2.09.01-4.42.05-7.04.03-2.74.12-5.12.25-7.28.13-2.09.36-3.94.71-5.47.23-1.07.61-1.61.61-1.61l2.2-2.92-12.7.03,2.46,2.94Z"/>
      <path class="st0" d="M870.83,182.95c-3-2.79-6.48-4.9-10.33-6.24-7.66-2.7-15.75-2.7-23.42,0-3.85,1.34-7.33,3.43-10.34,6.24-2.97,2.77-5.42,6.24-7.26,10.33-1.84,4.07-2.77,8.91-2.77,14.41s.93,10.28,2.77,14.37c1.84,4.11,4.29,7.59,7.27,10.34,2.98,2.73,6.45,4.82,10.3,6.22,3.86,1.38,7.82,2.07,11.72,2.07s7.88-.69,11.73-2.07c3.88-1.4,7.35-3.49,10.32-6.24,3-2.75,5.43-6.24,7.26-10.33,1.84-4.13,2.77-8.97,2.77-14.37s-.93-10.3-2.77-14.42c-1.82-4.06-4.26-7.54-7.26-10.32ZM869.14,218.06c-.77,3.99-2.07,7.35-3.85,9.99-1.77,2.61-3.94,4.63-6.47,5.98-4.59,2.47-9.71,2.69-14.82.56-2.26-.94-4.44-2.42-6.47-4.41-2-1.96-3.8-4.55-5.33-7.69-1.56-3.15-2.72-6.93-3.48-11.25-.92-5.2-1.02-9.88-.28-13.91.77-3.98,2.05-7.35,3.85-10,1.76-2.6,3.94-4.63,6.48-6.01,2.51-1.36,5.23-2.04,7.96-2.04,2.3,0,4.6.47,6.84,1.43,2.3.96,4.45,2.45,6.4,4.44,2.02,2.03,3.83,4.63,5.37,7.77,1.57,3.12,2.75,6.9,3.52,11.24.93,5.22,1.02,9.91.28,13.91Z"/>
      <path class="st0" d="M815.45,237.15c-.5-.23-2.29-1.06-3.86-2.04-2-1.3-3.95-2.76-5.8-4.32-2.07-1.72-4.09-3.59-6.01-5.57-1.99-2-3.78-3.98-5.32-5.89-1.54-1.87-2.77-3.57-3.78-5.21-1.11-1.8-1.23-2.67-1.23-2.94,0-.4.24-.78.68-1.19.85-.7,1.86-1.39,3.09-2.1,1.38-.77,2.86-1.63,4.4-2.56,1.64-.98,3.21-2.12,4.64-3.41,1.5-1.33,2.77-2.9,3.78-4.71,1.03-1.87,1.55-4.1,1.55-6.59,0-2.12-.46-4.18-1.37-6.09-.93-1.95-2.39-3.69-4.35-5.17-1.88-1.45-4.3-2.6-7.18-3.42-2.85-.82-6.26-1.23-10.12-1.23h-27.26s2.44,3.04,2.44,3.04l.05.05c.08.13.42.65.62,1.53.33,1.53.57,3.41.72,5.58.15,2.23.23,4.72.23,7.4,0,0,.07,28.46.04,31.08-.04,2.69-.12,5.15-.27,7.29-.15,2.21-.38,4.05-.72,5.44-.23.98-.62,1.51-.62,1.51l-2.76,3.08,19.46-.03-2.55-2.98c-.29-.42-.53-.98-.66-1.62-.36-1.47-.6-3.36-.73-5.57-.13-2.2-.23-4.63-.25-7.43-.04-2.67-.05-5.04-.05-7.18v-36.7h12.31c3.93,0,6.78,1.09,8.71,3.3,1.96,2.24,2.97,5.38,2.97,9.36,0,2.88-.61,5.24-1.79,7.03-1.3,1.96-2.76,3.69-4.34,5.13l-4.68,4.27c-1.8,1.66-2.72,3.62-2.72,5.81,0,1.78.49,3.82,1.54,6.26.92,2.18,2.2,4.48,3.79,6.88,1.66,2.43,3.47,4.75,5.41,6.91,2.12,2.36,4.31,4.43,6.54,6.17l.5.36h22.97l-8.05-3.53Z"/>
    </g>
    <path class="st0" d="M782.89,149.02c-5.81,0-11.84-1.33-16.58-3.62l-1.68-.82.67-1.43c7.06.72,3.83.72,15.34.61,15.61-.16,13.99-12.48,13.99-12.48h-14.93s-20.25.15-20.25-24.01c0-16.92,12.61-25.77,25.08-25.77s21.02,8.44,21.02,20.05v23.57h2.2c4.26,0,8.82-3.01,8.82-11.45v-9.92c0-13.1,9.4-22.26,22.87-22.26,11.23,0,23.24,6.39,23.24,24.36v7.81c0,8.55,2.96,11.45,11.7,11.45h48.61c-2.77-3.95-4.64-9.68-6.58-15.66-3.65-11.23-7.43-22.85-17.22-22.85-7.77,0-16.59,8.82-21.5,17.02l-3.73,6.21.49-18.4c9.27-7.72,18.53-9.93,24.35-9.93,20.77,0,25.62,15.74,29.52,28.38,2.92,9.44,4.99,15.24,10.58,15.24h4.56c4.26,0,8.82-3.01,8.82-11.45v-9.92c0-13.1,9.4-22.26,22.87-22.26,11.19,0,23.24,8.09,23.24,25.86,0,19.01-11.11,25.78-21.53,25.78-6.91,0-13.91-2.94-17.98-7.39-3.04,3.5-7.52,5.52-12.43,5.52h-74.73c-7.64,0-13.06-2.08-16.17-6.19-3.78,4.84-9.39,7.56-15.84,7.56s-12.75-2.68-16.58-6.87c-3.04,3.49-7.51,5.5-12.41,5.5h-5.49c0,12.55-9.04,17.75-22.35,17.75ZM975.16,86.34c-8.17,0-11.78,5.88-12.06,19.66l-.17,9.11c0,2.56-.7,5.45-1.88,7.69,2.63,1.86,7.93,4.56,12.76,4.56h0c2.53,0,6.14-.74,8.97-4.27,2.83-3.55,4.27-9.06,4.27-16.35,0-13.38-5.98-20.4-11.88-20.4ZM839.43,86.34c-8.17,0-11.78,5.88-12.06,19.66l-.19,9.11c0,2.57-.7,5.46-1.87,7.69,2.53,1.7,7.55,4.18,12.36,4.18h0c2.6,0,6.31-.74,9.23-4.27,2.92-3.53,4.4-8.97,4.4-16.14,0-13.9-6.17-20.23-11.88-20.23ZM783.02,86.34c-6.07,0-12.23,6.98-12.23,20.32s6.14,18.46,11.88,18.46h12.19l-.15-21.37c0-10.9-4.38-17.41-11.7-17.41ZM902.17,148.04c-3.37,0-6.11-2.77-6.11-6.19s2.8-6.11,6.11-6.11,6.1,2.75,6.1,6.11-2.73,6.19-6.1,6.19ZM693.5,131.27c-9.59,0-15.52-6.75-15.85-18.08v-24.77c-18.71,0-22.02,13.18-22.02,20.59,0,6.49,2.95,14.07,14.26,14.07h1.9v2.93h-4.89c-16.65,0-20.11-8.97-20.11-16.74,0-11.01,8.74-26.57,32.75-26.65h8.13v31.04c0,8.44,4.56,11.45,8.82,11.45h19.58c-4.91-4.54-7.61-11.15-7.61-18.91,0-16.11,12.33-24.8,23.94-24.8,13.97,0,20.65,5.89,20.65,5.89l-3.32,9.56-1.75-2.32c-3.08-4.12-9.27-8.55-15.58-8.55-8.22,0-12.94,6.54-12.94,17.95,0,10.27,5.96,21.19,16.98,21.19h10.57c6.8,0,6.89-.09,8.28-.56,1.39-.47,3.26-2.1,3.26-2.1l-1.34,8.82h-63.69ZM680.69,78.35c-3.37,0-6.11-2.79-6.11-6.21s2.8-6.1,6.11-6.1,6.1,2.73,6.1,6.1-2.73,6.21-6.1,6.21ZM662.25,78.35c-3.37,0-6.11-2.79-6.11-6.21s2.8-6.1,6.11-6.1,6.11,2.73,6.11,6.1-2.75,6.21-6.11,6.21Z"/>
    <g>
      <path class="st0" d="M608.05,82.64c.39.61.78,1.21,1.17,1.82.52.8,1.13,1.58,1.44,2.49.15.44.25.9.34,1.36.23,1.14.36,2.3.47,3.46.16,1.71.24,3.43.29,5.15v33.17c0,10.07-2.32,13.99-8.28,13.99-1.94,0-9.39-1.29-9.39-1.29l-1.66,3.25,1.86.81c3.69,1.59,7.47,2.43,10.94,2.43,10.93,0,17.44-7.18,17.44-19.19v-27.64c0-3.35-.01-6.7.05-10.05.03-1.87-.11-3.81.47-5.61.45-1.38,1.23-2.7,1.9-4.14h-17.05Z"/>
      <path class="st0" d="M590.11,125.24v-23.57c0-11.62-8.83-20.05-21.02-20.05s-25.08,8.85-25.08,25.77c0,24.16,20.25,24.01,20.25,24.01h14.93s1.62,12.32-13.99,12.48c-11.5.12-8.28.11-15.34-.61l-.67,1.43,1.68.82c4.73,2.29,10.77,3.62,16.58,3.62,13.31,0,22.35-5.2,22.35-17.75l.31-6.15ZM567.25,125.24c-5.74,0-11.88-4.85-11.88-18.46s6.15-20.32,12.23-20.32c7.32,0,11.7,6.51,11.7,17.41l.15,21.37h-12.19Z"/>
      <g>
        <path class="st0" d="M485.53,78.35c3.37,0,6.11-2.79,6.11-6.21s-2.75-6.1-6.11-6.1-6.1,2.79-6.1,6.1,2.73,6.21,6.1,6.21Z"/>
        <path class="st0" d="M497.85,72.14c0,3.42,2.75,6.21,6.11,6.21s6.11-2.79,6.11-6.21-2.75-6.1-6.11-6.1-6.11,2.79-6.11,6.1Z"/>
        <path class="st0" d="M494.67,64.65c3.37,0,6.1-2.77,6.1-6.19s-2.73-6.11-6.1-6.11-6.11,2.8-6.11,6.11,2.75,6.19,6.11,6.19Z"/>
        <path class="st0" d="M399.45,85.61c3.37,0,6.11-2.79,6.11-6.21s-2.75-6.1-6.11-6.1-6.1,2.79-6.1,6.1,2.73,6.21,6.1,6.21Z"/>
        <path class="st0" d="M519.62,86.67c.38.83.57,1.67.72,2.56.43,2.53.54,5.12.61,7.68v18.16c0,8.45-4.55,11.46-8.81,11.46-5.54,0-11.57-5.05-11.8-14.35v-5.7c0-3.22.04-6.45.22-9.67.1-1.72.16-3.47.64-5.14.42-1.46,1.26-2.78,1.94-4.13h-17.27c.27,0,1.4,2.14,1.62,2.47,1.35,2.05,1.54,4.52,1.73,6.9.25,3.13.31,6.28.32,10.18v7.98c0,8.45-4.55,11.46-8.81,11.46-5.54,0-11.57-5.05-11.8-16.51,0-4.87-.19-9.98.63-15.35.3-1.22,1.14-2.42,1.71-3.55h-16.9c.57.89,1.15,1.78,1.72,2.67,1.32,2.05,1.45,4.59,1.66,6.95.2,2.26.29,4.54.33,6.81.02,1.03.03,2.06.03,3.09v3.91c0,9.34-7.59,13.58-14.63,13.58s-13.81-3.67-14.51-11.71v-5.5c0-4.25.04-8.49.35-12.73.12-1.62.14-3.1.88-4.56.43-.84.85-1.68,1.28-2.52h-17.1c.62.96,1.27,1.92,1.87,2.89,1.93,3.1,1.73,7.85,1.83,11.38v16.27c0,14.89-6.99,22.15-21.37,22.15s-25.23-9.39-25.23-22.32c0-19.68,19.51-19.68,19.51-19.68l-8.36-9.32c-10.08,4.84-16.34,15.62-16.34,28.13,0,16.87,12.72,28.65,30.94,28.65s28.73-7.77,31.16-22.47c4.27,4.75,11.55,6.92,18.21,6.92,8.06,0,16.95-3.26,20.6-10.31,3.75,6.34,10.66,9.27,16.5,9.27s12.16-2.85,15.04-9.09c3.78,6.21,10.61,9.09,16.38,9.09,8.01,0,16.63-5.33,16.63-17.07l.09-1.96v-11.21c0-3.22.04-6.44.18-9.66.09-1.97.01-4.27.54-6.18.38-1.36,1.19-2.67,1.79-3.94h-17.05c.18,0,1.01,1.64,1.15,1.87.37.62.73,1.24,1.16,2.14Z"/>
      </g>
    </g>
  </g>
</svg>'''

try:
        import cairosvg
        _CAIROSVG_AVAILABLE = True
except ImportError:
        _CAIROSVG_AVAILABLE = False

# ============================================================================
# EMBEDDED ROSHN LOGO (Base64 encoded PNG)
# ============================================================================
ROSHN_LOGO_BASE64 = """
iVBORw0KGgoAAAANSUhEUgAAASwAAABGCAYAAABjhLnnAAAACXBIWXMAABYlAAAWJQFJUiTw
AAAWLklEQVR4nO2dCXgU1RnHv9nNJptsNhuS3CEhCSHcEO4rchQQRAUVrFprveu1rRet1qq1
WmuttVZba63W4lFEKwooIhAIORAI4QghJJBACOQkZJPsZrO7Sd73zXtfmExmNyEBxGWf5//s
zs7s7Lz5zf+97817IwgKhUKhUCgUCoVCoVAoFAqFQqFQKBQKhUKhUCgUCoVCoVAoFAqFQqFQ
KBQKhUKhUCgUCoVCoVAoFArFh5CC14+B14+C1/8avP5v4fX/A6//b/D6P4LX3wuvvxFe/yl4
/WXg9ReC1+8Fr28Cr68Dr78Qvf4kfHwiez3M2YFhEAIa+O8L0ev/Arz+DHj9n4HXzwGvnwBe
bwGvV8HrZ4HXjwevrwKv7w5eXwavXwNeHwKvl8LrSwWvNwuvdxe83i68vgS8vgd4fTK8vtX5
8/eB1y8Cr58FXr8UvP5FePwT4PX/Bq8/H7x+Gnh9T/B6NXh9M/D6aHj9QHh9PHh9J/B6Oby+
AL1+K/D6u8HrZ4LXz4bXXwmvnwOP/0F4/e/h9V+B15eC1/cHr1fA6yPB6wPg9RL4+lTw+lbg
9Vng9cPB65PB6zuD1zcHr/8AvP4ieP1l4PX/gMdXgdffCq9fBV5fBl4fCl4fD14fDl6vhK+3
AK8Xw+s7gdf3Bq8fBV6fBF7fCby+M3h9S3h9Gnh9LHh9DHh9V/D6RPD6DuD1AeD1nuD1XuD1
3cHr+4HXp4LX9wKvjwSv9wOvdwevdwGvdwCvF8HXm4LXNwevbwJe7wde7wtenw5e3xu8vh94
fVfw+u7g9THg9V3B6zuA13uB17uC17uA1zuD1zuB1zuB17uC1/uC1weC18eA17cHr28DXu8P
Xu8JXu8BXu8OXt8NvD4RvD4JvL4neH0P8PoI8PpA8Ho/8Hov8Po68PoO4PWdweuDwOu7gdfH
g9cngNfHg9d3A68PB6/vBF7fFrw+BLw+ALzeD7zeG7zeC7zeE7zeFbzeHbzeHbw+ArwuGV4f
B1wPA6+Pgq8Hgdd3BK/vCF7fEby+A3i9L3i9G3i9J3i9F3i9N3i9H3h9AHh9IHi9P3i9P3h9
IHh9AHi9P3i9P3h9IHh9EHh9MHh9CHh9Z/D6MPD6TuD1HcHrI8DrfcHrPcHrA8Dr/cDrfcHr
fcHrfcDrvcHrvcDr3cHrPcDrI8DrO4PXdwKv7wRe3wm8vhN4fSfw+o7g9R3B6zuC13cEr+8E
Xt8ZvL4zeH1H8Po48Pow8PpI8Ppw8PpI8PpI8Pow8Pow8Pou4PVdwOvDwevDwOu7gNeHgtd3
Bq/vDF7fBbw+FLw+DLy+C3h9F/D6LuD1ncHrO4PXdwGv7wxe3wW8Pgy8Pgy8vgt4fRh4fRfw
+i7g9WHg9Z3B6zuD14eC13cGrw8Fr+8CXt8FvD4MvL4zeH0X8Pou4PWdwevDwOu7gNeHgdeH
gtd3Bq8PBa/vAl4fBl4fCl7fGbw+DLy+M3h9GHh9Z/D6UPD6zuD1ncHrQ8Hrw8Dru4DXh4HX
dwGv7wxe3xm8Pgy8vjN4fRh4fWfw+jDw+s7g9WHg9aHg9Z3B68PA67uA14eB14eC14eC13cB
rw8Dr+8MXh8GXt8FvD4UvL4zeH0YeH1n8Pow8Pou4PVh4PWdweu7gNeHgteHgteHgdd3Aa8P
A6/vAl4fBl7fBbw+DLy+M3h9GHh9Z/D6MPD6LuD1YeD1XcDrw8Drw8DrO4PXh4HXdwGvDwOv
7wxeHwZe3wW8Pgy8vjN4fRh4fRh4fWfw+jDw+i7g9WHg9V3A68PA67uA14eBqwPA68PA6zuD
14eB13cBrw8Dr+8CXh8GXt8FvD4MvL4LeH0YeH0X8Pow8PrO4PVh4PVdwOvDwOu7gNeHgdd3
Aa8PA6/vAl4fBl7fBbw+DLy+C3h9GHh9F/D6MPD6LuD1YeD1XcDrw8Dru4DXh4HXdwGvDwOv
7wJeHwZe3wW8Pgy8vgt4fRh4fRfw+jDw+i7g9WHg9V3A68PA67uA14eB13cBrw8Dr+8CXh8G
Xt8FvD4MvL4LeH0YeP0/wOvDwOu7gNeHgdd3Aa8PA6/vAl4fBl7fBbw+DLw+HLx+Onh9OHh9
OHh9OHh9N/D6cPD6cPD6buD1IeD14eD13cDrw8Hrw8HrQ8Drw8DrQ8Drw8Hrw8HrQ8Drw8Hr
u4HXh4PXh4PXdwOvDwevDwev7wZeHw5eHw5e3w28Phy8Phy8vht4fTh4fTh4fTfw+nDw+nDw
+m7g9eHg9eHg9d3A68PB68PB68PB68PB68PB68PB68PB67uB14eD14eD14eD14eD14eD14eD
14eD14eD13cDrw8Hrw8Hr+8GXh8OXh8OXt8NvD4cvD4cvD4cvD4cvD4cvL4beH04eH04eH04
eH04eH04eH04eH04eH04eH04eH03+PoQ8Ppw8Ppw8PpQ8Ppw8Ppw8Ppw8Ppw8Ppw8Ppw8Ppu
4PXh4PXh4PXh4PXh4PXh4PXh4PXh4PXh4PXh4PXdwOvDwevDweu7gdeHg9eHg9eHg9eHg9eH
g9eHg9eHg9d3A68PB68PB6/vBl4fDl4fDl4fDl4fDl4fDl4fDl4fDl7fDbw+HLw+HLw+HLy+
G3h9OHh9OHh9OHh9OHh9OHh9OHh9OHh9N/D6cPD6cPD6cPD6buD14eD14eD14eD14eD14eD1
4eD14eD13cDrw8Hrw8Hrw8Hru4HXh4PXh4PXh4PXh4PXh4PXh4PXh4PXdwOvDwevDwevDwev
7wZeHw5eHw5eHw5eHw5eHw5eHw5eHw5eHw5e3w28Phy8Phy8Phy8Phy8vht4fTh4fTh4fTh4
fTh4fTh4fTh4fTh4fTfw+nDw+nDw+nDw+nDw+m7g9eHg9eHg9eHg9eHg9eHg9eHg9eHg9eHg
9d3A68PB68PB68PB68PB67uB14eD14eD14eD14eD14eD14eD14eD14eD13cDrw8Hrw8Hrw8H
rw8Hr+8GXh8OXh8OXh8OXh8OXh8OXh8OXh8OXh8OXt8NvD4cvD4cvD4cvD4cvD4cvL4beH04
eH04eH04eH04eH04eH04eH04eH04eH038Ppw8Ppw8Ppw8Ppw8Ppw8Ppu4PXh4PXh4PXh4PXh
4PXh4PXh4PXh4PXh4PXdwOvDwevDwevDwevDwevDweu7gdeHg9eHg9eHg9eHg9eHg9eHg9eH
g9eHg9d3A68PB68PB68PB68PB68PB6/vBl4fDl4fDl4fDl4fDl4fDl4fDl4fDl4fDl7fDby+
C3h9OHh9OHh9OHh9OHh9OHh9OHh9N/D6cPD6cPD6cPD6cPD6cPD6cPD6cPD6buD14eD14eD1
4eD14eD14eD14eD14eDrz4fXh4PXh4PXh4PXh4PXh4PXh4PXh4PXh4PXd4Ov3wivDwevDwev
Dwevb8HX9wevDwevDwevDwevDwevDwev7wZfvwh4fTh4fTh4fTh4fTh4fQi8fjV4fTh4fTh4
fTh4fTh4fTh4fTfw+nDw+nDw+nDw+nDw+nDw+nDw+l7g9eHg9eHg9eHg9eHg9eHg9eHg9eHg
9d3A68PB68PB68PB68PB68PB68PB68PB67uB14eD14eD14eD14eD14eD14eD14eD14eDrzfA
60PA68PB68PB68PB68PB68PB68PB67uB14eD14eD14eD14eD14eD14eD14eD14eD1/tXgdeH
g9eHg9eHg9eHg9eHg9eHg9eHg9eHg9d3Ba8PB68PB68PB68PB68PB68PB68PB68PB6+/DbK+
B3h9OHh9OHh9OHh9OHh9OHh9OHh9OHh9OHh9MXh9OHh9OHh9OHh9OHh9OHh9OHh9OHh9OHh9
Onh9A3h9OHh9OHh9OHh9OHh9OHh9OHh9OHh9OHh9A3h9OHh9OHh9OHh9OHh9OHh9OHh9OHh9
OHj9b8Drw8Hrw8Hrw8Hrw8Hrw8Hrw8Hrw8Hrw8Hrbwdvfxh4fTh4fTh4fTh4fTh4fTh4fTh4
fTh4fTh4/a3g7Q8Drw8Hrw8Hrw8Hrw8Hrw8Hrw8Hrw8Hrw8Hr78JvP1h4PXh4PXh4PXh4PXh
4PXh4PXh4PXh4PXh4PU3grc/DLw+HLw+HLw+HLw+HLw+HLw+HLw+HLw+HLz+OvD2h4HXh4PX
h4PXh4PXh4PXh4PXh4PXh4PXh4PXXwPe/jDw+nDw+nDw+nDw+nDw+nDw+nDw+nDw+nDw+n+B
tz8MvD4cvD4cvD4cvD4cvD4cvD4cvD4cvD4cvP6f4O0PA68PB68PB68PB68PB68PB68PB68P
B68PB6//O3j7w8Drw8Hrw8Hrw8Hrw8Hrw8Hrw8Hrw8Hrw8HrbwBvfxh4fTh4fTh4fTh4fTh4
fTh4fTh4fTh4fTh4fRl4+8PA68PB68PB68PB68PB68PB68PB68PB68PB6y8Fb38YeH04eH04
eH04eH04eH04eH04eH04eH04eP0F4O0PA68PB68PB68PB68PB68PB68PB68PB68PB6//E3j7
w8Drw8Hrw8Hrw8Hrw8Hrw8Hrw8Hrw8Hrw8HrfwOv/y24+sPA68PB68PB68PB68PB68PB68PB
68PB68PB6y8BV38YeH04eH04eH04eH04eH04eH04eH04eH04eP0vwdUfBl4fDl4fDl4fDl4f
Dl4fDl4fDl4fDl4fDl7/c3D1h4HXh4PXh4PXh4PXh4PXh4PXh4PXh4PXh4PX/xRc/WHg9eHg
9eHg9eHg9eHg9eHg9eHg9eHg9eHg9f8HV38YeH04eH04eH04eH04eH04eH04eH04eH04eP0P
wdUfBl4fDl4fDl4fDl4fDl4fDl4fDl4fDl4fDl7/fXD1h4HXh4PXh4PXh4PXh4PXh4PXh4PX
h4PXh4PXfxdc/WHg9eHg9eHg9eHg9eHg9eHg9eHg9eHg9eHg9ReBqz8MvD4cvD4cvD4cvD4c
vD4cvD4cvD4cvD4cvP58cPWHgdeHg9eHg9eHg9eHg9eHg9eHg9eHg9eHg9dfCK7+MPD6cPD6
cPD6cPD6cPD6cPD6cPD6cPD6cPD688DVHwZeHw5eHw5eHw5eHw5eHw5eHw5eHw5eHw5e/xNw
9YeB14eD14eD14eD14eD14eD14eD14eD14eD158Drv4w8Ppw8Ppw8Ppw8Ppw8Ppw8Ppw8Ppw
8Ppw8PqfgKs/DLw+HLw+HLw+HLw+HLw+HLw+HLw+HLw+HLz+HHD1h4HXh4PXh4PXh4PXh4PX
h4PXh4PXh4PXh4PXnwOu/jDw+nDw+nDw+nDw+nDw+nDw+nDw+nDw+nDw+nPA1R8GXh8OXh8O
Xh8OXh8OXh8OXh8OXh8OXh8OXn8OuPrDwOvDwevDwevDwevDwevDwevDwevDwevDwevPAVd/
GHh9OHh9OHh9OHh9OHh9OHh9OHh9OHh9OHj9OeDqDwOvDwevDwevDwevDwevDwevDwevDwev
Dwev/wm4+sPA68PB68PB68PB68PB68PB68PB68PB68PB688BV38YeH04eH04eH04eH04eH04
eH04eH04eH04eP054OoPA68PB68PB68PB68PB68PB68PB68PB68PB68/B1z9YeD14eD14eD1
4eD14eD14eD14eD14eD14eD154CrPwy8Phy8Phy8Phy8Phy8Phy8Phy8Phy8Phy8/hxw9YeB
14eD14eD14eD14eD14eD14eD14eD14eD1/8EXP1h4PXh4PXh4PXh4PXh4PXh4PXh4PXh4PXh
4PXngKs/DLw+HLw+HLw+HLw+HLw+HLw+HLw+HLw+HLz+HHD1h4HXh4PXh4PXh4PXh4PXh4PX
h4PXh4PXh4PXnwOu/jDw+nDw+nDw+nDw+nDw+nDw+nDw+nDw+nDw+nPA1R8GXh8OXh8OXh8O
Xh8OXh8OXh8OXh8OXh8OXn8OuPrDwOvDwevDwevDwevDwevDwevDwevDwevDwevPAVd/GHh9
OHh9OHh9OHh9OHh9OHh9OHh9OHh9OHj9T8DVHwZeHw5eHw5eHw5eHw5eHw5eHw5eHw5eHw5e
fw64+sPA68PB68PB68PB68PB68PB68PB68PB68PB688BV38Y8wlX74dhPL/+CPP4x5nHv8J8
/j4m8g8xnm9jJv8xzObPYAb/TybxN5vNO8Kl9cdYTb4D5vFNsIn/PBbwr8VKvhgb+BjYyJfD
Zr4ctvJPwHb+FdjJPxwP8M/Dbv65eJC/E/bwpXiYv8Z8viWO8tdYwL+GOXxX2MOPYDEfhyX8
KJbxXbCcb4sVfAfkchXI51tjFV+OJ/hKrOY/gtV8N6zh07CWz8R6PgUb+F+bzZfCJv4dZvMV
sIXvhq38aOTxCdhuNq+C7XwqcvlkPMQXYwdfBgV8BRTy2VDEn4YSvg3K+FQo50NRwYdBJR+P
R/gIVPEJUM0n4VG+APL5cNTwiajhU7COD0cdn4r1fDrW8xnYwGdiI5+Fx/g8bOJzsJnPxhN8
Lp7kM7CFz8JWPhvb+Cxs5zOxg0/HLj4DO/lU7OJTsJtPwh7+cuzlL8E+Phn7+cfhAL8YD/Ln
4yD/OMzhq+AQ/3kc5v+AI/wJOMJnYx//GTjK18JR/u44xv8ejvPF8Dj/ZjzBX4on+bPxFF8C
T/PXwjP89fAsfxM8x98Mz/O3wAv8bfACfzu8yN8BL/F3wsu8G17h74JXeQ94jb8bXudr4Q2+
Ft7kU/AWfwLeVp7gFb4a3uFL4H1eCR/wSviQL4YP+RR8xBfCx3wmfMKfhE/5h+Az/h34nP8q
fMGfgy/50/AlXwVf8a3xNV8I3/AF8C1fDN/xhfA9fxL+xy+Ek3wBnOIvgdN8CZzhm+AMXw2n
+cr/5/N5/tCfxzF4/p+8D8Pz/X6e4PV+Hs8P/E88oT+4w0+8j8PvO/x7ns+4jjG/53g/h+f5
gee5kz/h+V7+Hp7P5/c8oP/WE+r4BmXsm57Q//8Jz/fw73meL/Rvev7Qj/V4wv//8ISe7+Wv
ex5wuGXIkn3BExr/Hk+o8Yd6Qj9w8HhCp3zVE3p+Qj+fBxzxO+D0h57Q8yd/xOHxhP7+8gxZ
gM99nkb+Bb/oB//gF/5Yvxc8QzwnPKH//hOe0PO9/M3/h9/1BM6XuU54Qp/38m96Qs9f/Hs9
oedf9i89IS/4FU/ouYPH+znE86s88+D3f8Gj4e94wuNR+f6f+Ht4wuN//YTH4/mf9HhC39/v
8SRY57l6jid0e8njcXhO/oTnZ/y+xO+7PZ4l/y1/BxjzO8/z/V5gPN/v9znEBR9hPOE/X/pT
n/CEfuQ0R3zCE/qxe8Lj8YTOU/AET+gvnOEJnfNxjyf0Y5f5vMTv+zwe4QntOfAjPKFHO3t+
1hN61+EOz/sxwdOPcHie/ypP6JkOno8Jnk/4gid0m3/AJ3z8e/Jre0IcBz0ej+c3/YQn9By/
6wk9/ZL/5Qkd/gNP6Nl3/wZP6P+DPt8hXq/nh+56Qq9JhnqCT/4fHvD5BE/oJ/30v3E80e+X
eEJb3eMJPfvyVz2hz5/8dU9ouDM8Id/4qif0ZzeH5we9v/WE/sLB4wnNfd0Tejynnv8kT2jc
pzy+l4A4fH7de7Q75/M5yLMhvN7j8D7hCT3n4NeHP/MjnAfFE/rAR33CF/y5J5Tz4AlP6C+b
eELnJfiaD53jhAe8/m883u8TnF/29b3rCQ/n9/lcgcU+v+bx+Dz/p3k8j/8Pj9fzGU+ozR73
BAV9ixdq+FIB7+kBMIVCoVAoFAqFQqFQKBQKhUKhUCgUCoVCoVAoFAqFQqFQKBQKhUKhUCgU
CoVCoVAoFAqFQqFQKBQKhUKhUCgUCoVCoVAoFAqFQqFQKBQKhUKhUCgUCoVCoVAoFAqFQqFQ
KBQKhUKhUCgUCoVCoVAoFAqFQqFQKBQKhUKhUCgUCoVCoVAoFAqFQqFQKBQKhUKhUCgUCoVC
oVAoFAqFQqFQKBQKhUKhUCgUCoVCoVAoFAqFQqFQKBQKhUKhUCgUCoVCoVAoFAqFQqFQKBQK
hUKhUCgUCoVCoVAoFAqFQqFQKBQKhUL5/8B/AQUGqKGsrcwdAAAAAElFTkSuQmCC
"""


# ============================================================================
# ROSHN BRAND COLORS & THEME
# ============================================================================
class ROSHNTheme:
    """ROSHN Brand Colors and Styling - Based on Official Branding"""
    
    # Primary Colors - ROSHN Teal/Green
    PRIMARY = "#006B5A"           # ROSHN Primary Teal Green
    PRIMARY_LIGHT = "#008B74"     # Lighter Teal
    PRIMARY_DARK = "#004D40"      # Darker Teal
    
    # Secondary/Accent - Complementary colors
    ACCENT = "#00A88E"            # Bright Teal (for highlights)
    ACCENT_LIGHT = "#4DB6A4"      # Light Teal
    ACCENT_DARK = "#007A68"       # Dark Teal
    
    # Background Colors
    BG_DARK = "#004D40"           # Dark teal background
    BG_MAIN = "#F5F7F6"           # Light gray-green background
    BG_CARD = "#FFFFFF"           # White for cards
    BG_SECONDARY = "#E8F0EE"      # Secondary background (light teal tint)
    
    # Text Colors
    TEXT_PRIMARY = "#1A1A1A"      # Near black text
    TEXT_SECONDARY = "#5A6A68"    # Gray-teal text
    TEXT_LIGHT = "#FFFFFF"        # White text
    TEXT_MUTED = "#8A9A98"        # Muted teal-gray text
    
    # Status Colors
    SUCCESS = "#28A745"           # Green
    SUCCESS_LIGHT = "#D4EDDA"     # Light green background
    WARNING = "#F5A623"           # Amber/Orange
    WARNING_LIGHT = "#FFF3CD"     # Light yellow background
    ERROR = "#DC3545"             # Red
    ERROR_LIGHT = "#F8D7DA"       # Light red background
    INFO = "#17A2B8"              # Cyan/Teal
    INFO_LIGHT = "#D1ECF1"        # Light cyan background
    
    # ROSHN Specific
    ROSHN_FLOWER = "#006B5A"      # Flower pattern color
    ROSHN_DARK = "#004D40"        # Dark variant
    
    # Fonts
    FONT_FAMILY = "Segoe UI"
    FONT_FAMILY_ARABIC = "Arial"
    FONT_SIZE_TITLE = 24
    FONT_SIZE_HEADER = 16
    FONT_SIZE_NORMAL = 11
    FONT_SIZE_SMALL = 9
    
    # Dimensions
    BUTTON_WIDTH = 20
    ENTRY_WIDTH = 50
    PADDING = 10
    CARD_PADDING = 15
    BORDER_RADIUS = 8


# ============================================================================
# CUSTOM WIDGETS
# ============================================================================
class ModernButton(tk.Button):
    """Modern styled button with hover effects"""
    
    def __init__(self, parent, text, command=None, style="primary", **kwargs):
        self.style = style
        colors = self._get_colors()
        
        super().__init__(
            parent,
            text=text,
            command=command,
            font=(ROSHNTheme.FONT_FAMILY, ROSHNTheme.FONT_SIZE_NORMAL, "bold"),
            bg=colors["bg"],
            fg=colors["fg"],
            activebackground=colors["active_bg"],
            activeforeground=colors["active_fg"],
            relief="flat",
            cursor="hand2",
            padx=20,
            pady=10,
            **kwargs
        )
        
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
    
    def _get_colors(self):
        if self.style == "primary":
            return {
                "bg": ROSHNTheme.PRIMARY,
                "fg": ROSHNTheme.TEXT_LIGHT,
                "active_bg": ROSHNTheme.PRIMARY_LIGHT,
                "active_fg": ROSHNTheme.TEXT_LIGHT,
                "hover_bg": ROSHNTheme.PRIMARY_LIGHT
            }
        elif self.style == "accent":
            return {
                "bg": ROSHNTheme.ACCENT,
                "fg": ROSHNTheme.TEXT_PRIMARY,
                "active_bg": ROSHNTheme.ACCENT_LIGHT,
                "active_fg": ROSHNTheme.TEXT_PRIMARY,
                "hover_bg": ROSHNTheme.ACCENT_LIGHT
            }
        elif self.style == "secondary":
            return {
                "bg": ROSHNTheme.BG_SECONDARY,
                "fg": ROSHNTheme.TEXT_PRIMARY,
                "active_bg": ROSHNTheme.BG_CARD,
                "active_fg": ROSHNTheme.TEXT_PRIMARY,
                "hover_bg": "#D0D0D0"
            }
        elif self.style == "success":
            return {
                "bg": ROSHNTheme.SUCCESS,
                "fg": ROSHNTheme.TEXT_LIGHT,
                "active_bg": "#218838",
                "active_fg": ROSHNTheme.TEXT_LIGHT,
                "hover_bg": "#218838"
            }
        else:  # outline
            return {
                "bg": ROSHNTheme.BG_CARD,
                "fg": ROSHNTheme.PRIMARY,
                "active_bg": ROSHNTheme.PRIMARY,
                "active_fg": ROSHNTheme.TEXT_LIGHT,
                "hover_bg": ROSHNTheme.PRIMARY_LIGHT
            }
    
    def _on_enter(self, event):
        colors = self._get_colors()
        self.configure(bg=colors["hover_bg"])
    
    def _on_leave(self, event):
        colors = self._get_colors()
        self.configure(bg=colors["bg"])


class CardFrame(tk.Frame):
    """Card-style frame with shadow effect"""
    
    def __init__(self, parent, title=None, **kwargs):
        super().__init__(
            parent,
            bg=ROSHNTheme.BG_CARD,
            highlightbackground=ROSHNTheme.BG_SECONDARY,
            highlightthickness=1,
            **kwargs
        )
        
        if title:
            title_label = tk.Label(
                self,
                text=title,
                font=(ROSHNTheme.FONT_FAMILY, ROSHNTheme.FONT_SIZE_HEADER, "bold"),
                bg=ROSHNTheme.BG_CARD,
                fg=ROSHNTheme.PRIMARY,
                anchor="w"
            )
            title_label.pack(fill="x", padx=ROSHNTheme.CARD_PADDING, pady=(ROSHNTheme.CARD_PADDING, 5))
            
            # Separator line - ROSHN Teal accent
            sep = tk.Frame(self, height=2, bg=ROSHNTheme.PRIMARY)
            sep.pack(fill="x", padx=ROSHNTheme.CARD_PADDING, pady=(0, 10))


class StatusBadge(tk.Label):
    """Status badge widget"""
    
    def __init__(self, parent, text="", status="info", **kwargs):
        colors = self._get_status_colors(status)
        
        super().__init__(
            parent,
            text=text,
            font=(ROSHNTheme.FONT_FAMILY, ROSHNTheme.FONT_SIZE_SMALL, "bold"),
            bg=colors["bg"],
            fg=colors["fg"],
            padx=10,
            pady=3,
            **kwargs
        )
    
    def _get_status_colors(self, status):
        if status == "success":
            return {"bg": ROSHNTheme.SUCCESS_LIGHT, "fg": ROSHNTheme.SUCCESS}
        elif status == "warning":
            return {"bg": ROSHNTheme.WARNING_LIGHT, "fg": "#856404"}
        elif status == "error":
            return {"bg": ROSHNTheme.ERROR_LIGHT, "fg": ROSHNTheme.ERROR}
        else:
            return {"bg": ROSHNTheme.INFO_LIGHT, "fg": ROSHNTheme.INFO}
    
    def set_status(self, text, status):
        colors = self._get_status_colors(status)
        self.configure(text=text, bg=colors["bg"], fg=colors["fg"])


class ToolTip:
    """Tooltip widget for hover help"""
    
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.show_delay = 500  # milliseconds
        self.scheduled_show = None
        
        widget.bind("<Enter>", self.schedule_show, add="+")
        widget.bind("<Leave>", self.hide, add="+")
    
    def schedule_show(self, event=None):
        # Cancel any pending show
        if self.scheduled_show:
            self.widget.after_cancel(self.scheduled_show)
            self.scheduled_show = None
        
        # Schedule show with delay
        if not self.tooltip:
            self.scheduled_show = self.widget.after(self.show_delay, self.show)
    
    def show(self, event=None):
        self.scheduled_show = None
        
        if self.tooltip:
            return
        
        # Get widget position
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        
        # Create tooltip window
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        # CRITICAL: Make tooltip completely non-interactive and click-through
        self.tooltip.attributes('-topmost', True)
        
        # Platform-specific click-through settings
        try:
            # For Windows - disable mouse events
            self.tooltip.attributes('-transparentcolor', self.tooltip['bg'])
            self.tooltip.attributes('-alpha', 0.95)
        except:
            pass
        
        try:
            # For macOS
            self.tooltip.attributes('-transparent', True)
        except:
            pass
        
        label = tk.Label(
            self.tooltip,
            text=self.text,
            font=(ROSHNTheme.FONT_FAMILY, ROSHNTheme.FONT_SIZE_SMALL),
            bg="#FFFFD0",
            fg=ROSHNTheme.TEXT_PRIMARY,
            relief="solid",
            borderwidth=1,
            padx=5,
            pady=3
        )
        label.pack()
        
        # Prevent tooltip from stealing focus
        self.tooltip.update_idletasks()
    
    def hide(self, event=None):
        # Cancel any scheduled show
        if self.scheduled_show:
            self.widget.after_cancel(self.scheduled_show)
            self.scheduled_show = None
        
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


# ============================================================================
# QUEUE HANDLER FOR LOGGING
# ============================================================================
class QueueHandler(logging.Handler):
    """Logging handler that sends log records to a queue"""
    
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
    
    def emit(self, record):
        self.log_queue.put(record)


# ============================================================================
# MAIN APPLICATION
# ============================================================================
class PDFValidationApp:
    """Main PDF Validation Application"""
    
    # AI Cost per file (approximate USD)
    AI_COSTS = {
        "claude": 0.015,      # Claude Vision ~$0.015 per image/page analyzed
        "gemini": 0.005,      # Gemini Vision ~$0.005 per image
        "openai": 0.020,      # GPT-4 Vision ~$0.02 per image
        "auto": 0.015,        # Auto uses Claude primarily
    }
    
    # Pages analyzed with AI per PDF (facade detection mainly)
    AI_PAGES_PER_PDF = 3  # Pages 3, 8, 9 may use Vision
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Roshn Demarcation Plan Validation System")
        self.root.geometry("1200x850")
        self.root.minsize(1000, 700)
        self.root.configure(bg=ROSHNTheme.BG_MAIN)
        
        # Set icon if available
        try:
            # self.root.iconbitmap("roshn_icon.ico")
            pass
        except:
            pass
        
        # Variables
        self.pdf_path = tk.StringVar()
        self.excel_path = tk.StringVar(value="./input_data/ground_truth.xlsx")
        self.output_dir = tk.StringVar(value="./output")
        self.use_ai = tk.BooleanVar(value=True)
        self.ai_model = tk.StringVar(value="auto")
        self.is_running = False
        self.validation_thread = None
        
        # Queue for log messages
        self.log_queue = queue.Queue()
        
        # Results storage
        self.results = []
        self.total_cost = 0.0
        
        # Setup UI
        self._setup_styles()
        self._create_header()
        self._create_main_content()
        self._create_footer()
        
        # Start log polling
        self._poll_log_queue()
    
    def _setup_styles(self):
        """Configure ttk styles with ROSHN theme"""
        style = ttk.Style()
        style.theme_use("clam")
        
        # Configure progress bar - ROSHN Teal
        style.configure(
            "Custom.Horizontal.TProgressbar",
            troughcolor=ROSHNTheme.BG_SECONDARY,
            background=ROSHNTheme.PRIMARY,
            lightcolor=ROSHNTheme.ACCENT,
            darkcolor=ROSHNTheme.PRIMARY_DARK,
            bordercolor=ROSHNTheme.BG_SECONDARY,
            thickness=20
        )
        
        # Configure checkbutton
        style.configure(
            "Custom.TCheckbutton",
            background=ROSHNTheme.BG_CARD,
            foreground=ROSHNTheme.TEXT_PRIMARY,
            font=(ROSHNTheme.FONT_FAMILY, ROSHNTheme.FONT_SIZE_NORMAL)
        )
        
        # Configure radiobutton
        style.configure(
            "Custom.TRadiobutton",
            background=ROSHNTheme.BG_CARD,
            foreground=ROSHNTheme.TEXT_PRIMARY,
            font=(ROSHNTheme.FONT_FAMILY, ROSHNTheme.FONT_SIZE_NORMAL)
        )
    
    def _create_header(self):
        """Create application header with ROSHN branding"""
        header = tk.Frame(self.root, bg=ROSHNTheme.BG_CARD, height=90)
        header.pack(fill="x")
        header.pack_propagate(False)

        # Logo area - use embedded ROSHN logo
        logo_frame = tk.Frame(header, bg=ROSHNTheme.BG_CARD)
        logo_frame.pack(side="left", padx=20, pady=15)

        # Load and display embedded ROSHN logo (SVG rendered to PNG)
        try:
            if _CAIROSVG_AVAILABLE:
                logo_height = 50
                png_bytes = cairosvg.svg2png(bytestring=ROSHN_LOGO_SVG.encode('utf-8'), output_height=logo_height)
                logo_image = Image.open(BytesIO(png_bytes))
                self.logo_photo = ImageTk.PhotoImage(logo_image)
                logo_label = tk.Label(
                    logo_frame,
                    image=self.logo_photo,
                    bg=ROSHNTheme.BG_CARD
                )
                logo_label.pack(side="left", padx=(0, 20))
            else:
                raise RuntimeError("cairosvg not available")
        except Exception as e:
            # Fallback to text if logo fails
            tk.Label(
                logo_frame,
                text="ROSHN",
                font=(ROSHNTheme.FONT_FAMILY, 20, "bold"),
                bg=ROSHNTheme.BG_CARD,
                fg=ROSHNTheme.PRIMARY
            ).pack(side="left", padx=(0, 20))

        # Separator line
        sep = tk.Frame(logo_frame, width=2, bg=ROSHNTheme.PRIMARY)
        sep.pack(side="left", fill="y", pady=5)

        # Application titles
        app_frame = tk.Frame(logo_frame, bg=ROSHNTheme.BG_CARD)
        app_frame.pack(side="left", padx=(20, 0))

        tk.Label(
            app_frame,
            text="Roshn Demarcation Plan Validation System",
            font=(ROSHNTheme.FONT_FAMILY, 16, "bold"),
            bg=ROSHNTheme.BG_CARD,
            fg=ROSHNTheme.PRIMARY
        ).pack(anchor="w")

        tk.Label(
            app_frame,
            text="Automated Document Verification",
            font=(ROSHNTheme.FONT_FAMILY, 10),
            bg=ROSHNTheme.BG_CARD,
            fg=ROSHNTheme.TEXT_SECONDARY
        ).pack(anchor="w")

        # Version badge on right
        version_frame = tk.Frame(header, bg=ROSHNTheme.PRIMARY)
        version_frame.pack(side="right", padx=20, pady=25)
        tk.Label(
            version_frame,
            text=" v1.0 ",
            font=(ROSHNTheme.FONT_FAMILY, 9, "bold"),
            bg=ROSHNTheme.PRIMARY,
            fg=ROSHNTheme.TEXT_LIGHT,
        ).pack(padx=4, pady=2)

        # Bottom border line - FULL WIDTH
        border = tk.Frame(self.root, height=3, bg=ROSHNTheme.PRIMARY)
        border.pack(fill="x")
    
    def _create_main_content(self):
        """Create main content area with scrolling support"""
        # Create container for canvas and scrollbar
        container = tk.Frame(self.root, bg=ROSHNTheme.BG_MAIN)
        container.pack(fill="both", expand=True)

        # Create canvas
        self.canvas = tk.Canvas(container, bg=ROSHNTheme.BG_MAIN, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        # Create scrollbar
        scrollbar = tk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")

        # Configure canvas
        self.canvas.configure(yscrollcommand=scrollbar.set)

        # Create main frame inside canvas
        main = tk.Frame(self.canvas, bg=ROSHNTheme.BG_MAIN)
        self.canvas_window = self.canvas.create_window((0, 0), window=main, anchor="nw")

        # Bind canvas resize
        main.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Enable mousewheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Add padding to main content
        content_frame = tk.Frame(main, bg=ROSHNTheme.BG_MAIN)
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Create two-column layout with equal heights
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)

        left_panel = tk.Frame(content_frame, bg=ROSHNTheme.BG_MAIN)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        right_panel = tk.Frame(content_frame, bg=ROSHNTheme.BG_MAIN)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        # Left panel - Input configuration
        self._create_input_card(left_panel)
        self._create_ai_options_card(left_panel)
        self._create_actions_card(left_panel)

        # Right panel - Output and results
        self._create_progress_card(right_panel)
        self._create_results_card(right_panel)

    def _on_canvas_configure(self, event):
        """Update canvas window width when canvas is resized"""
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def _create_input_card(self, parent):
        """Create input files card"""
        card = CardFrame(parent, title="📁 Input Files")
        card.pack(fill="x", pady=(0, 15))
        
        content = tk.Frame(card, bg=ROSHNTheme.BG_CARD)
        content.pack(fill="x", padx=ROSHNTheme.CARD_PADDING, pady=(0, ROSHNTheme.CARD_PADDING))
        
        # PDF Input
        pdf_frame = tk.Frame(content, bg=ROSHNTheme.BG_CARD)
        pdf_frame.pack(fill="x", pady=5)
        
        tk.Label(
            pdf_frame,
            text="PDF File or Folder:",
            font=(ROSHNTheme.FONT_FAMILY, ROSHNTheme.FONT_SIZE_NORMAL),
            bg=ROSHNTheme.BG_CARD,
            fg=ROSHNTheme.TEXT_PRIMARY
        ).pack(anchor="w")
        
        pdf_input_frame = tk.Frame(pdf_frame, bg=ROSHNTheme.BG_CARD)
        pdf_input_frame.pack(fill="x", pady=5)
        
        self.pdf_entry = tk.Entry(
            pdf_input_frame,
            textvariable=self.pdf_path,
            font=(ROSHNTheme.FONT_FAMILY, ROSHNTheme.FONT_SIZE_NORMAL),
            bg=ROSHNTheme.BG_CARD,
            fg=ROSHNTheme.TEXT_PRIMARY,
            relief="solid",
            borderwidth=1
        )
        self.pdf_entry.pack(side="left", fill="x", expand=True, ipady=8)
        
        ModernButton(
            pdf_input_frame, text="📄 File", command=self._browse_pdf_file,
            style="secondary", width=8
        ).pack(side="left", padx=(5, 0))
        
        ModernButton(
            pdf_input_frame, text="📂 Folder", command=self._browse_pdf_folder,
            style="secondary", width=8
        ).pack(side="left", padx=(5, 0))
        
        # Excel Input
        excel_frame = tk.Frame(content, bg=ROSHNTheme.BG_CARD)
        excel_frame.pack(fill="x", pady=5)
        
        tk.Label(
            excel_frame,
            text="Ground Truth Excel:",
            font=(ROSHNTheme.FONT_FAMILY, ROSHNTheme.FONT_SIZE_NORMAL),
            bg=ROSHNTheme.BG_CARD,
            fg=ROSHNTheme.TEXT_PRIMARY
        ).pack(anchor="w")
        
        excel_input_frame = tk.Frame(excel_frame, bg=ROSHNTheme.BG_CARD)
        excel_input_frame.pack(fill="x", pady=5)
        
        self.excel_entry = tk.Entry(
            excel_input_frame,
            textvariable=self.excel_path,
            font=(ROSHNTheme.FONT_FAMILY, ROSHNTheme.FONT_SIZE_NORMAL),
            bg=ROSHNTheme.BG_CARD,
            fg=ROSHNTheme.TEXT_PRIMARY,
            relief="solid",
            borderwidth=1
        )
        self.excel_entry.pack(side="left", fill="x", expand=True, ipady=8)
        
        ModernButton(
            excel_input_frame, text="Browse", command=self._browse_excel,
            style="secondary", width=10
        ).pack(side="left", padx=(5, 0))
        
        # Output Directory
        output_frame = tk.Frame(content, bg=ROSHNTheme.BG_CARD)
        output_frame.pack(fill="x", pady=5)
        
        tk.Label(
            output_frame,
            text="Output Directory:",
            font=(ROSHNTheme.FONT_FAMILY, ROSHNTheme.FONT_SIZE_NORMAL),
            bg=ROSHNTheme.BG_CARD,
            fg=ROSHNTheme.TEXT_PRIMARY
        ).pack(anchor="w")
        
        output_input_frame = tk.Frame(output_frame, bg=ROSHNTheme.BG_CARD)
        output_input_frame.pack(fill="x", pady=5)
        
        self.output_entry = tk.Entry(
            output_input_frame,
            textvariable=self.output_dir,
            font=(ROSHNTheme.FONT_FAMILY, ROSHNTheme.FONT_SIZE_NORMAL),
            bg=ROSHNTheme.BG_CARD,
            fg=ROSHNTheme.TEXT_PRIMARY,
            relief="solid",
            borderwidth=1
        )
        self.output_entry.pack(side="left", fill="x", expand=True, ipady=8)
        
        ModernButton(
            output_input_frame, text="Browse", command=self._browse_output,
            style="secondary", width=10
        ).pack(side="left", padx=(5, 0))
    
    def _create_ai_options_card(self, parent):
        """Create AI options card"""
        card = CardFrame(parent, title="🤖 AI Analysis Options")
        card.pack(fill="x", pady=(0, 15))
        
        content = tk.Frame(card, bg=ROSHNTheme.BG_CARD)
        content.pack(fill="x", padx=ROSHNTheme.CARD_PADDING, pady=(0, ROSHNTheme.CARD_PADDING))
        
        # AI Toggle
        toggle_frame = tk.Frame(content, bg=ROSHNTheme.BG_CARD)
        toggle_frame.pack(fill="x", pady=5)
        
        self.ai_check = ttk.Checkbutton(
            toggle_frame,
            text="Enable AI Vision Analysis (Facade Detection & Dimension Verification)",
            variable=self.use_ai,
            style="Custom.TCheckbutton",
            command=self._update_ai_options
        )
        self.ai_check.pack(anchor="w")
        
        # AI Info
        ai_info_frame = tk.Frame(content, bg=ROSHNTheme.INFO_LIGHT)
        ai_info_frame.pack(fill="x", pady=10)
        
        tk.Label(
            ai_info_frame,
            text="ℹ️  AI Vision is used for facade style detection (Traditional/Modern) and dimension verification when text extraction is insufficient.",
            font=(ROSHNTheme.FONT_FAMILY, ROSHNTheme.FONT_SIZE_SMALL),
            bg=ROSHNTheme.INFO_LIGHT,
            fg=ROSHNTheme.INFO,
            wraplength=400,
            justify="left"
        ).pack(padx=10, pady=8)
        
        # AI Model Selection (initially hidden, shown via _update_ai_options)
        self.model_frame = tk.LabelFrame(
            content,
            text="AI Model",
            font=(ROSHNTheme.FONT_FAMILY, ROSHNTheme.FONT_SIZE_NORMAL),
            bg=ROSHNTheme.BG_CARD,
            fg=ROSHNTheme.TEXT_PRIMARY
        )
        # Don't pack initially - let _update_ai_options handle it

        models = [
            ("Auto (Recommended)", "auto", "$0.015/file"),
            ("Claude Vision", "claude", "$0.015/file"),
            ("Gemini Vision", "gemini", "$0.005/file"),
            ("OpenAI GPT-4 Vision", "openai", "$0.020/file")
        ]

        for model_name, model_value, cost in models:
            frame = tk.Frame(self.model_frame, bg=ROSHNTheme.BG_CARD)
            frame.pack(anchor="w", padx=10, pady=2)

            ttk.Radiobutton(
                frame,
                text=model_name,
                variable=self.ai_model,
                value=model_value,
                style="Custom.TRadiobutton",
                command=self._update_cost_estimate
            ).pack(side="left")

            tk.Label(
                frame,
                text=cost,
                font=(ROSHNTheme.FONT_FAMILY, ROSHNTheme.FONT_SIZE_SMALL),
                bg=ROSHNTheme.BG_CARD,
                fg=ROSHNTheme.TEXT_MUTED
            ).pack(side="left", padx=(10, 0))

        # Cost Estimate (initially hidden, shown via _update_ai_options)
        self.cost_frame = tk.Frame(content, bg=ROSHNTheme.WARNING_LIGHT)
        # Don't pack initially - let _update_ai_options handle it

        self.cost_label = tk.Label(
            self.cost_frame,
            text="💰 Estimated AI Cost: $0.00 (0 files)",
            font=(ROSHNTheme.FONT_FAMILY, ROSHNTheme.FONT_SIZE_NORMAL, "bold"),
            bg=ROSHNTheme.WARNING_LIGHT,
            fg="#856404"
        )
        self.cost_label.pack(padx=10, pady=8)

        # Initialize visibility based on checkbox state
        self._update_ai_options()
    
    def _create_actions_card(self, parent):
        """Create actions card"""
        card = CardFrame(parent, title="⚡ Actions")
        card.pack(fill="x", pady=(0, 15))
        
        content = tk.Frame(card, bg=ROSHNTheme.BG_CARD)
        content.pack(fill="x", padx=ROSHNTheme.CARD_PADDING, pady=(0, ROSHNTheme.CARD_PADDING))
        
        # Buttons row
        btn_frame = tk.Frame(content, bg=ROSHNTheme.BG_CARD)
        btn_frame.pack(fill="x", pady=10)
        
        self.start_btn = ModernButton(
            btn_frame,
            text="▶️  Start Validation",
            command=self._start_validation,
            style="success",
            width=18
        )
        self.start_btn.pack(side="left", padx=(0, 10))
        ToolTip(self.start_btn, "Start validating PDFs against ground truth Excel")
        
        self.stop_btn = ModernButton(
            btn_frame,
            text="⏹️  Stop",
            command=self._stop_validation,
            style="secondary",
            width=12
        )
        self.stop_btn.pack(side="left", padx=(0, 10))
        self.stop_btn.configure(state="disabled")
        ToolTip(self.stop_btn, "Stop the current validation process")
        
        open_btn = ModernButton(
            btn_frame,
            text="📂 Open Output",
            command=self._open_output_folder,
            style="outline",
            width=14
        )
        open_btn.pack(side="left")
        ToolTip(open_btn, "Open the output folder in File Explorer")
    
    def _create_progress_card(self, parent):
        """Create progress and log card"""
        card = CardFrame(parent, title="📊 Progress & Log")
        card.pack(fill="x", pady=(0, 15))
        
        content = tk.Frame(card, bg=ROSHNTheme.BG_CARD)
        content.pack(fill="both", expand=True, padx=ROSHNTheme.CARD_PADDING, pady=(0, ROSHNTheme.CARD_PADDING))
        
        # Status bar
        status_frame = tk.Frame(content, bg=ROSHNTheme.BG_CARD)
        status_frame.pack(fill="x", pady=(0, 10))
        
        self.status_badge = StatusBadge(status_frame, text="Ready", status="info")
        self.status_badge.pack(side="left")
        
        # Clear log button
        clear_btn = tk.Button(
            status_frame,
            text="🗑️ Clear Log",
            font=(ROSHNTheme.FONT_FAMILY, ROSHNTheme.FONT_SIZE_SMALL),
            command=self._clear_log,
            bg=ROSHNTheme.BG_SECONDARY,
            fg=ROSHNTheme.TEXT_PRIMARY,
            relief="flat",
            cursor="hand2",
            padx=8,
            pady=2
        )
        clear_btn.pack(side="right")
        
        self.file_count_label = tk.Label(
            status_frame,
            text="0 / 0 files",
            font=(ROSHNTheme.FONT_FAMILY, ROSHNTheme.FONT_SIZE_NORMAL),
            bg=ROSHNTheme.BG_CARD,
            fg=ROSHNTheme.TEXT_SECONDARY
        )
        self.file_count_label.pack(side="right", padx=(0, 10))
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            content,
            variable=self.progress_var,
            maximum=100,
            style="Custom.Horizontal.TProgressbar"
        )
        self.progress_bar.pack(fill="x", pady=(0, 10))
        
        # Log output
        log_frame = tk.Frame(content, bg=ROSHNTheme.BG_CARD)
        log_frame.pack(fill="both", expand=True)
        
        self.log_text = ScrolledText(
            log_frame,
            font=("Consolas", 9),
            bg="#1E1E1E",
            fg="#D4D4D4",
            insertbackground="#D4D4D4",
            relief="flat",
            wrap="word"
        )
        self.log_text.pack(fill="both", expand=True)
        
        # Configure log text tags for colors
        self.log_text.tag_configure("INFO", foreground="#4EC9B0")
        self.log_text.tag_configure("WARNING", foreground="#DCDCAA")
        self.log_text.tag_configure("ERROR", foreground="#F14C4C")
        self.log_text.tag_configure("SUCCESS", foreground="#6A9955")
        self.log_text.tag_configure("TIMESTAMP", foreground="#808080")
    
    def _clear_log(self):
        """Clear the log text"""
        self.log_text.delete(1.0, tk.END)
    
    def _create_results_card(self, parent):
        """Create results summary card"""
        card = CardFrame(parent, title="📈 Results Summary")
        card.pack(fill="x")
        
        content = tk.Frame(card, bg=ROSHNTheme.BG_CARD)
        content.pack(fill="x", padx=ROSHNTheme.CARD_PADDING, pady=(0, ROSHNTheme.CARD_PADDING))
        
        # Stats grid
        stats_frame = tk.Frame(content, bg=ROSHNTheme.BG_CARD)
        stats_frame.pack(fill="x", pady=10)
        
        # Row 1
        row1 = tk.Frame(stats_frame, bg=ROSHNTheme.BG_CARD)
        row1.pack(fill="x", pady=5)
        
        self._create_stat_box(row1, "PDFs Validated", "0", "pdfs_validated")
        self._create_stat_box(row1, "Total Checks", "0", "total_checks")
        self._create_stat_box(row1, "Match Rate", "0%", "match_rate", highlight=True)
        
        # Row 2
        row2 = tk.Frame(stats_frame, bg=ROSHNTheme.BG_CARD)
        row2.pack(fill="x", pady=5)
        
        self._create_stat_box(row2, "Matches", "0", "matches", color=ROSHNTheme.SUCCESS)
        self._create_stat_box(row2, "Mismatches", "0", "mismatches", color=ROSHNTheme.ERROR)
        self._create_stat_box(row2, "AI Cost", "$0.00", "ai_cost", color=ROSHNTheme.INFO)
        
        # Report location
        self.report_frame = tk.Frame(content, bg=ROSHNTheme.SUCCESS_LIGHT)
        self.report_frame.pack(fill="x", pady=10)
        self.report_frame.pack_forget()  # Hide initially
        
        self.report_label = tk.Label(
            self.report_frame,
            text="📄 Report: ",
            font=(ROSHNTheme.FONT_FAMILY, ROSHNTheme.FONT_SIZE_NORMAL),
            bg=ROSHNTheme.SUCCESS_LIGHT,
            fg=ROSHNTheme.SUCCESS,
            cursor="hand2"
        )
        self.report_label.pack(padx=10, pady=8)
        self.report_label.bind("<Button-1>", self._open_report)
    
    def _create_stat_box(self, parent, label, value, var_name, color=None, highlight=False):
        """Create a stat display box"""
        frame = tk.Frame(parent, bg=ROSHNTheme.BG_SECONDARY if not highlight else ROSHNTheme.ACCENT)
        frame.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        
        bg_color = ROSHNTheme.BG_SECONDARY if not highlight else ROSHNTheme.ACCENT
        fg_color = color or ROSHNTheme.TEXT_PRIMARY if not highlight else ROSHNTheme.TEXT_LIGHT
        
        tk.Label(
            frame,
            text=label,
            font=(ROSHNTheme.FONT_FAMILY, ROSHNTheme.FONT_SIZE_SMALL),
            bg=bg_color,
            fg=ROSHNTheme.TEXT_SECONDARY if not highlight else ROSHNTheme.TEXT_LIGHT
        ).pack(pady=(8, 0))
        
        value_label = tk.Label(
            frame,
            text=value,
            font=(ROSHNTheme.FONT_FAMILY, 18, "bold"),
            bg=bg_color,
            fg=fg_color
        )
        value_label.pack(pady=(0, 8))
        
        # Store reference to update later
        setattr(self, f"stat_{var_name}", value_label)
    
    def _create_footer(self):
        """Create application footer with ROSHN branding"""
        footer = tk.Frame(self.root, bg=ROSHNTheme.PRIMARY_DARK, height=35)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        
        tk.Label(
            footer,
            text="© 2025 ROSHN Group | مجموعة روشن | Roshn Demarcation Plan Validation System | Automated Document Verification",
            font=(ROSHNTheme.FONT_FAMILY, ROSHNTheme.FONT_SIZE_SMALL),
            bg=ROSHNTheme.PRIMARY_DARK,
            fg=ROSHNTheme.ACCENT_LIGHT
        ).pack(pady=8)
    
    # ========================================================================
    # FILE BROWSING
    # ========================================================================
    def _browse_pdf_file(self):
        """Browse for a single PDF file"""
        filename = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )
        if filename:
            self.pdf_path.set(filename)
            self._update_cost_estimate()

    def _browse_pdf_folder(self):
        """Browse for a folder containing PDFs"""
        folder = filedialog.askdirectory(title="Select PDF Folder")
        if folder:
            self.pdf_path.set(folder)
            self._update_cost_estimate()
    
    def _browse_excel(self):
        """Browse for Excel ground truth file"""
        filename = filedialog.askopenfilename(
            title="Select Ground Truth Excel",
            filetypes=[("Excel Files", "*.xlsx *.xls"), ("All Files", "*.*")]
        )
        if filename:
            self.excel_path.set(filename)
    
    def _browse_output(self):
        """Browse for output directory"""
        folder = filedialog.askdirectory(title="Select Output Directory")
        if folder:
            self.output_dir.set(folder)
    
    def _open_output_folder(self):
        """Open output folder in file explorer"""
        output_path = self.output_dir.get()
        if os.path.exists(output_path):
            os.startfile(output_path)
        else:
            messagebox.showwarning("Warning", "Output folder does not exist yet.")
    
    def _open_report(self, event=None):
        """Open the generated report"""
        if hasattr(self, 'report_path') and self.report_path:
            os.startfile(self.report_path)
    
    # ========================================================================
    # AI OPTIONS
    # ========================================================================
    def _update_ai_options(self):
        """Update AI options visibility based on toggle"""
        if self.use_ai.get():
            self.model_frame.pack(fill="x", pady=5)
            self.cost_frame.pack(fill="x", pady=10)
        else:
            self.model_frame.pack_forget()
            self.cost_frame.pack_forget()
        self._update_cost_estimate()
    
    def _update_cost_estimate(self):
        """Update cost estimate based on file count and AI settings"""
        if not self.use_ai.get():
            self.cost_label.configure(text="💰 AI Disabled - No Cost")
            return
        
        # Count PDF files
        pdf_path = self.pdf_path.get()
        file_count = 0
        
        if pdf_path:
            if os.path.isfile(pdf_path) and pdf_path.lower().endswith('.pdf'):
                file_count = 1
            elif os.path.isdir(pdf_path):
                file_count = len(list(Path(pdf_path).glob("*.pdf")))
        
        # Calculate cost
        model = self.ai_model.get()
        cost_per_file = self.AI_COSTS.get(model, 0.015)
        total_cost = file_count * cost_per_file
        
        self.cost_label.configure(
            text=f"💰 Estimated AI Cost: ${total_cost:.2f} ({file_count} files × ${cost_per_file:.3f})"
        )
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    def _start_validation(self):
        """Start the validation process"""
        # Validate inputs
        pdf_path = self.pdf_path.get()
        excel_path = self.excel_path.get()
        output_dir = self.output_dir.get()
        
        if not pdf_path:
            messagebox.showerror("Error", "Please select a PDF file or folder.")
            return
        
        if not os.path.exists(pdf_path):
            messagebox.showerror("Error", f"PDF path does not exist: {pdf_path}")
            return
        
        if not excel_path or not os.path.exists(excel_path):
            messagebox.showerror("Error", f"Ground truth Excel file not found: {excel_path}")
            return
        
        # Update UI state
        self.is_running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_badge.set_status("Running...", "warning")
        self.progress_var.set(0)
        self.log_text.delete(1.0, tk.END)
        
        # Reset results
        self.results = []
        self.total_cost = 0.0
        self.report_frame.pack_forget()
        
        # Start validation in background thread
        self.validation_thread = threading.Thread(
            target=self._run_validation,
            args=(pdf_path, excel_path, output_dir),
            daemon=True
        )
        self.validation_thread.start()
    
    def _stop_validation(self):
        """Stop the validation process"""
        self.is_running = False
        self._log_message("Stopping validation...", "WARNING")
        self.status_badge.set_status("Stopping...", "warning")
    
    def _run_validation(self, pdf_path: str, excel_path: str, output_dir: str):
        """Run validation in background thread"""
        try:
            # Setup logging to capture messages
            queue_handler = QueueHandler(self.log_queue)
            queue_handler.setFormatter(logging.Formatter('%(message)s'))
            
            # Import validation modules
            from pdf_extractor import PDFExtractor
            from excel_loader import GroundTruthLoader
            from validator import Validator, ValidationReport
            from report_generator import ReportGenerator
            
            # Configure PDF extractor logging
            pdf_logger = logging.getLogger('pdf_extractor')
            pdf_logger.addHandler(queue_handler)
            pdf_logger.setLevel(logging.INFO)
            
            self._log_message("=" * 60, "INFO")
            self._log_message("ROSHN DEMARCATION PLAN VALIDATION SYSTEM", "SUCCESS")
            self._log_message("=" * 60, "INFO")
            
            # Load ground truth
            self._log_message(f"Loading ground truth from: {excel_path}", "INFO")
            gt_loader = GroundTruthLoader(excel_path)
            ground_truth = gt_loader.load()
            self._log_message(f"Loaded {len(ground_truth)} ground truth records", "SUCCESS")
            
            # Get PDF files
            if os.path.isfile(pdf_path):
                pdf_files = [pdf_path]
            else:
                pdf_files = list(Path(pdf_path).glob("*.pdf"))
            
            total_files = len(pdf_files)
            self._log_message(f"Found {total_files} PDF files to validate", "INFO")
            self.root.after(0, lambda: self.file_count_label.configure(text=f"0 / {total_files} files"))
            
            # Initialize validator and results
            validator = Validator()
            results = []
            ai_model = self.ai_model.get() if self.use_ai.get() else None
            
            # Process each PDF
            for i, pdf_file in enumerate(pdf_files):
                if not self.is_running:
                    self._log_message("Validation stopped by user", "WARNING")
                    break
                
                pdf_file = str(pdf_file)
                self._log_message(f"\n{'='*60}", "INFO")
                self._log_message(f"[{i+1}/{total_files}] Validating: {os.path.basename(pdf_file)}", "INFO")
                self._log_message(f"{'='*60}", "INFO")
                
                try:
                    # Extract data from PDF
                    debug_dir = os.path.join("./debug_images", Path(pdf_file).stem)
                    extractor = PDFExtractor(pdf_file, debug_dir, vision_model=ai_model or "auto")
                    
                    # Disable AI if requested
                    if not self.use_ai.get():
                        extractor.vision_model = None
                    
                    pdf_data = extractor.extract_all()
                    
                    # Find matching ground truth
                    gt_record = None
                    if pdf_data.plot_number:
                        gt_record = ground_truth.get(pdf_data.plot_number)
                    
                    if not gt_record and pdf_data.unit_code:
                        for record in ground_truth.values():
                            if record.unit_code == pdf_data.unit_code:
                                gt_record = record
                                break
                    
                    if gt_record:
                        # Validate
                        report = validator.validate(pdf_data, gt_record)
                        results.append(report)
                        
                        status = "✅ PASSED" if "PASSED" in report.overall_status else "❌ FAILED"
                        self._log_message(f"Result: {status} - {report.matches}/{report.total_checks} matches", 
                                         "SUCCESS" if "PASSED" in report.overall_status else "ERROR")
                        
                        # Update AI cost
                        if self.use_ai.get():
                            cost = self.AI_COSTS.get(ai_model or "auto", 0.015)
                            self.total_cost += cost
                    else:
                        self._log_message(f"No ground truth found for plot {pdf_data.plot_number}", "WARNING")
                
                except Exception as e:
                    self._log_message(f"Error processing {pdf_file}: {str(e)}", "ERROR")
                
                # Update progress
                progress = ((i + 1) / total_files) * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                self.root.after(0, lambda idx=i+1, total=total_files: 
                               self.file_count_label.configure(text=f"{idx} / {total} files"))
            
            # Generate report
            if results:
                self._log_message(f"\nGenerating reports...", "INFO")
                os.makedirs(output_dir, exist_ok=True)
                report_gen = ReportGenerator(output_dir)
                paths = report_gen.generate_all_reports(results, "validation")
                self.report_path = paths.get("excel", "")
                self._log_message(f"Report saved: {self.report_path}", "SUCCESS")
            
            # Store results
            self.results = results
            
            # Update UI with final results
            self.root.after(0, self._update_final_results)
            
        except Exception as e:
            self._log_message(f"Validation failed: {str(e)}", "ERROR")
            import traceback
            self._log_message(traceback.format_exc(), "ERROR")
        
        finally:
            self.root.after(0, self._validation_complete)
    
    def _validation_complete(self):
        """Called when validation is complete"""
        self.is_running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        
        if self.results:
            self.status_badge.set_status("Complete", "success")
        else:
            self.status_badge.set_status("No Results", "warning")
    
    def _update_final_results(self):
        """Update results summary"""
        if not self.results:
            return
        
        total_pdfs = len(self.results)
        total_checks = sum(r.total_checks for r in self.results)
        total_matches = sum(r.matches for r in self.results)
        total_mismatches = sum(r.mismatches for r in self.results)
        match_rate = (100 * total_matches / total_checks) if total_checks else 0
        
        # Update stat boxes
        self.stat_pdfs_validated.configure(text=str(total_pdfs))
        self.stat_total_checks.configure(text=str(total_checks))
        self.stat_match_rate.configure(text=f"{match_rate:.1f}%")
        self.stat_matches.configure(text=str(total_matches))
        self.stat_mismatches.configure(text=str(total_mismatches))
        self.stat_ai_cost.configure(text=f"${self.total_cost:.2f}")
        
        # Show report link
        if hasattr(self, 'report_path') and self.report_path:
            self.report_label.configure(text=f"📄 Report: {os.path.basename(self.report_path)} (Click to open)")
            self.report_frame.pack(fill="x", pady=10)
    
    def _log_message(self, message: str, level: str = "INFO"):
        """Log a message to the log panel"""
        self.root.after(0, lambda: self._append_log(message, level))
    
    def _append_log(self, message: str, level: str):
        """Append message to log text widget"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] ", "TIMESTAMP")
        self.log_text.insert(tk.END, f"{message}\n", level)
        self.log_text.see(tk.END)
    
    def _poll_log_queue(self):
        """Poll the log queue for new messages"""
        while True:
            try:
                record = self.log_queue.get_nowait()
                level = record.levelname
                message = record.getMessage()
                self._append_log(message, level)
            except queue.Empty:
                break
        
        self.root.after(100, self._poll_log_queue)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    """Main entry point for GUI application"""
    root = tk.Tk()
    
    # Hide main window initially for splash effect
    root.withdraw()
    
    # Show splash screen
    splash = tk.Toplevel()
    splash.overrideredirect(True)
    splash.configure(bg=ROSHNTheme.BG_CARD)
    
    # Center splash screen
    splash_width = 450
    splash_height = 320
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - splash_width) // 2
    y = (screen_height - splash_height) // 2
    splash.geometry(f"{splash_width}x{splash_height}+{x}+{y}")
    
    # Load and display embedded ROSHN logo on splash (SVG rendered to PNG)
    try:
        if _CAIROSVG_AVAILABLE:
            logo_height = 80
            png_bytes = cairosvg.svg2png(bytestring=ROSHN_LOGO_SVG.encode('utf-8'), output_height=logo_height)
            logo_image = Image.open(BytesIO(png_bytes))
            splash_logo_photo = ImageTk.PhotoImage(logo_image)
            logo_label = tk.Label(
                splash,
                image=splash_logo_photo,
                bg=ROSHNTheme.BG_CARD
            )
            logo_label.image = splash_logo_photo  # Keep reference
            logo_label.pack(pady=(40, 20))
        else:
            raise RuntimeError("cairosvg not available")
    except Exception as e:
        # Fallback to text if logo fails
        tk.Label(
            splash,
            text="ROSHN GROUP",
            font=(ROSHNTheme.FONT_FAMILY, 28, "bold"),
            bg=ROSHNTheme.BG_CARD,
            fg=ROSHNTheme.PRIMARY
        ).pack(pady=(40, 20))
    
    # Separator line
    sep_frame = tk.Frame(splash, bg=ROSHNTheme.PRIMARY, height=2, width=250)
    sep_frame.pack(pady=10)
    
    tk.Label(
        splash,
        text="Roshn Demarcation Plan Validation System",
        font=(ROSHNTheme.FONT_FAMILY, 16, "bold"),
        bg=ROSHNTheme.BG_CARD,
        fg=ROSHNTheme.PRIMARY
    ).pack(pady=(15, 5))
    
    tk.Label(
        splash,
        text="Loading...",
        font=(ROSHNTheme.FONT_FAMILY, 10),
        bg=ROSHNTheme.BG_CARD,
        fg=ROSHNTheme.TEXT_SECONDARY
    ).pack(pady=(15, 0))
    
    # Progress bar on splash
    style = ttk.Style()
    style.configure(
        "Splash.Horizontal.TProgressbar",
        troughcolor=ROSHNTheme.BG_SECONDARY,
        background=ROSHNTheme.PRIMARY,
        lightcolor=ROSHNTheme.ACCENT,
        darkcolor=ROSHNTheme.PRIMARY_DARK,
    )
    
    splash_progress = ttk.Progressbar(
        splash,
        length=300,
        mode='indeterminate',
        style="Splash.Horizontal.TProgressbar"
    )
    splash_progress.pack(pady=15)
    splash_progress.start(10)
    
    tk.Label(
        splash,
        text="© 2025 ROSHN Group | مجموعة روشن",
        font=(ROSHNTheme.FONT_FAMILY, 9),
        bg=ROSHNTheme.BG_CARD,
        fg=ROSHNTheme.TEXT_MUTED
    ).pack(side="bottom", pady=15)
    
    splash.update()
    
    # Simulate loading time
    root.after(1500, lambda: finish_loading(root, splash))
    
    root.mainloop()


def finish_loading(root, splash):
    """Finish loading and show main window"""
    splash.destroy()
    
    # Center window on screen
    window_width = 1200
    window_height = 850
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    app = PDFValidationApp(root)
    root.deiconify()


if __name__ == "__main__":
    main()
