Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public class TaskbarTransparency {
    [DllImport("user32.dll", SetLastError = true)]
    static extern IntPtr FindWindow(string lpClassName, string lpWindowName);

    [DllImport("dwmapi.dll")]
    public static extern int DwmSetWindowAttribute(IntPtr hwnd, int dwAttribute, ref int pvAttribute, int cbAttribute);

    public static void SetTransparent() {
        IntPtr taskbar = FindWindow("Shell_TrayWnd", null);
        if (taskbar != IntPtr.Zero) {
            int accent = 2;  // 2 = Full Transparency
            DwmSetWindowAttribute(taskbar, 19, ref accent, 4);
        }
    }
}
"@ -Language CSharp -PassThru | Out-Null

[TaskbarTransparency]::SetTransparent()