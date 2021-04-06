/*******************************************************************************
 * Copyright (c) 2021 Eurotech and/or its affiliates and others
 *
 * This program and the accompanying materials are made
 * available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 *
 * SPDX-License-Identifier: EPL-2.0
 *
 * Contributors:
 *  Eurotech
 *******************************************************************************/
package org.eclipse.kura.core.keystore;

import static java.util.Objects.isNull;

import java.util.Arrays;
import java.util.Map;

import org.eclipse.kura.configuration.Password;

public class KeystoreServiceOptions {

    private static final String KEY_SERVICE_PID = "kura.service.pid";
    static final String KEY_KEYSTORE_PATH = "keystore.path";
    static final String KEY_KEYSTORE_PASSWORD = "keystore.password";
    static final String KEY_RANDOMIZE_PASSWORD = "randomize.password";

    private static final String DEFAULT_KEYSTORE_PATH = "/tmp";
    private static final boolean DEFAULT_RANDOMIZE_PASSWORD = false;
    static final String DEFAULT_KEYSTORE_PASSWORD = "changeit";

    private final Map<String, Object> properties;
    private final String pid;
    private final String keystorePath;
    private final Password keystorePassword;
    private final boolean randomPassword;

    public KeystoreServiceOptions(Map<String, Object> properties) {
        if (isNull(properties)) {
            throw new IllegalArgumentException("Input parameters cannot be null!");
        }

        this.properties = properties;

        this.pid = (String) properties.get(KEY_SERVICE_PID);

        this.keystorePath = (String) properties.getOrDefault(KEY_KEYSTORE_PATH, DEFAULT_KEYSTORE_PATH);

        this.keystorePassword = new Password(
                (String) properties.getOrDefault(KEY_KEYSTORE_PASSWORD, DEFAULT_KEYSTORE_PASSWORD));

        this.randomPassword = (boolean) properties.getOrDefault(KEY_RANDOMIZE_PASSWORD, DEFAULT_RANDOMIZE_PASSWORD);
    }

    public Map<String, Object> getProperties() {
        return this.properties;
    }

    public String getPid() {
        return this.pid;
    }

    public String getKeystorePath() {
        return this.keystorePath;
    }

    public char[] getKeystorePassword() {
        return this.keystorePassword.getPassword();
    }

    public boolean needsRandomPassword() {
        return this.randomPassword;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + (this.keystorePassword == null ? 0 : this.keystorePassword.hashCode());
        result = prime * result + (this.keystorePath == null ? 0 : this.keystorePath.hashCode());
        result = prime * result + (this.pid == null ? 0 : this.pid.hashCode());
        result = prime * result + (this.properties == null ? 0 : this.properties.hashCode());
        result = prime * result + (this.randomPassword ? 1231 : 1237);
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        KeystoreServiceOptions other = (KeystoreServiceOptions) obj;
        if (this.keystorePassword == null) {
            if (other.keystorePassword != null) {
                return false;
            }
        } else if (!Arrays.equals(this.keystorePassword.getPassword(), other.keystorePassword.getPassword())) {
            return false;
        }
        if (this.keystorePath == null) {
            if (other.keystorePath != null) {
                return false;
            }
        } else if (!this.keystorePath.equals(other.keystorePath)) {
            return false;
        }
        if (this.pid == null) {
            if (other.pid != null) {
                return false;
            }
        } else if (!this.pid.equals(other.pid)) {
            return false;
        }
        if (this.properties == null) {
            if (other.properties != null) {
                return false;
            }
        } else if (!this.properties.equals(other.properties)) {
            return false;
        }
        boolean result = true;
        if (this.randomPassword != other.randomPassword) {
            result = false;
        }
        return result;
    }
}