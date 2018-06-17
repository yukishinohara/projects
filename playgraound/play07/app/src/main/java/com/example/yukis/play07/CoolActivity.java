package com.example.yukis.play07;

import android.content.ComponentName;
import android.content.Intent;
import android.content.ServiceConnection;
import android.os.IBinder;
import android.os.RemoteException;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Fuck;
import android.util.Log;
import android.view.View;

import com.example.apis.ICoolService;

public class CoolActivity extends AppCompatActivity {
    private final static String LOG_TAG = CoolActivity.class.getSimpleName();

    private ICoolService coolService = null;
    private final ServiceConnection coolServiceConnector = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName componentName, IBinder iBinder) {
            coolService = ICoolService.Stub.asInterface(iBinder);
        }

        @Override
        public void onServiceDisconnected(ComponentName componentName) {
            coolService = null;
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_cool);

        Intent intent = new Intent("com.example.coolservice.CoolService");
        intent.setPackage("com.example.coolservice");
        boolean success = bindService(intent, coolServiceConnector, BIND_AUTO_CREATE);
        Log.e(LOG_TAG, "### = " + success);
        Log.q(LOG_TAG, "hoge");
        Fuck.you(0xbabe);
    }

    @Override
    protected void onDestroy() {
        unbindService(coolServiceConnector);
        super.onDestroy();
    }

    public void onButtonClick(View view) {
        Log.e(LOG_TAG, "Kita.");
        try {
            coolService.printLogForTest("BAKA YAROU!!!!!!");
        } catch (RemoteException e) {
            e.printStackTrace();
        }
    }
}
