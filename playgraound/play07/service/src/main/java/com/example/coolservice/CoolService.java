package com.example.coolservice;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.os.RemoteException;
import android.support.annotation.Nullable;
import android.util.Log;

import com.example.apis.ICoolService;

/**
 * Created by yukis on 2017/06/15.
 */

public class CoolService extends Service {
    private final static String LOG_TAG = CoolService.class.getSimpleName();

    private final IBinder thisService = new ICoolService.Stub() {
        @Override
        public void printLogForTest(String message) throws RemoteException {
            Log.e(LOG_TAG, "" + message);
        }
    };

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        Log.e(LOG_TAG, "Bound.");
        return thisService;
    }
}
