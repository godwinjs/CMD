using System;
using System.Runtime.InteropServices;

class Program {
    [DllImport("user32.dll", SetLastError = true)]
    static extern IntPtr FindWindow(string lpClassName, string lpWindowName);

    [DllImport("dwmapi.dll")]
    public static extern int DwmSetWindowAttribute(IntPtr hwnd, int dwAttribute, ref int pvAttribute, int cbAttribute);

    static void Main() {
        IntPtr taskbar = FindWindow("Shell_TrayWnd", null);
        if (taskbar != IntPtr.Zero) {
            int accent = 2;  // 2 = Full Transparency
            DwmSetWindowAttribute(taskbar, 19, ref accent, 4);
        }
    }
}
