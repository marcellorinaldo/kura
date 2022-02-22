/*******************************************************************************
 * Copyright (c) 2022 Eurotech and/or its affiliates and others
 *
 * This program and the accompanying materials are made
 * available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 *
 * SPDX-License-Identifier: EPL-2.0
 *
 * Contributors:
 *  Eurotech
 ******************************************************************************/

package org.eclipse.kura.wire.ai.component.provider;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.eclipse.kura.KuraException;
import org.eclipse.kura.KuraIOException;
import org.eclipse.kura.ai.inference.Tensor;
import org.eclipse.kura.ai.inference.TensorDescriptor;
import org.eclipse.kura.type.BooleanValue;
import org.eclipse.kura.type.ByteArrayValue;
import org.eclipse.kura.type.DoubleValue;
import org.eclipse.kura.type.FloatValue;
import org.eclipse.kura.type.IntegerValue;
import org.eclipse.kura.type.LongValue;
import org.eclipse.kura.type.StringValue;
import org.eclipse.kura.type.TypedValue;
import org.eclipse.kura.wire.WireRecord;
import org.junit.Before;
import org.junit.Test;

public class TensorListAdapterTest {

    private Map<String, TypedValue<?>> wireRecordProperties;
    private WireRecord inputRecord;

    private List<Tensor> inputTensors;
    private List<TensorDescriptor> inputDescriptors;

    private List<Tensor> outputTensors;
    private List<WireRecord> outputRecords;

    private boolean exceptionOccurred = false;

    /*
     * Scenarios
     */
    @Test
    public void adapterShouldWorkWithBooleanScalarWiredRecord() {
        givenWireRecordPropWith("INPUT0", new BooleanValue(true));
        givenWireRecord();

        givenTensorDescriptorWith("INPUT0", "BOOL", Arrays.asList(1L, 1L));

        whenTensorListAdapterConvertsFromWireRecord();

        thenNoExceptionOccurred();
        thenResultingTensorListIsSize(1);
        thenResultingNamedTensorIsEqualTo("INPUT0", Boolean.class, Arrays.asList(true));
    }

    @Test
    public void adapterShouldWorkWithByteArrayWiredRecord() {
        givenWireRecordPropWith("INPUT0", new ByteArrayValue(new byte[] { 1, 2, 3, 4 }));
        givenWireRecord();

        givenTensorDescriptorWith("INPUT0", "BYTES", Arrays.asList(1L, 1L));

        whenTensorListAdapterConvertsFromWireRecord();

        thenNoExceptionOccurred();
        thenResultingTensorListIsSize(1);
        thenResultingNamedTensorIsEqualTo("INPUT0", Byte.class, Arrays.asList((byte) 1, (byte) 2, (byte) 3, (byte) 4));
    }

    @Test
    public void adapterShouldWorkWithFloatScalarWiredRecord() {
        givenWireRecordPropWith("INPUT0", new FloatValue(1.0F));
        givenWireRecord();

        givenTensorDescriptorWith("INPUT0", "FP32", Arrays.asList(1L, 1L));

        whenTensorListAdapterConvertsFromWireRecord();

        thenNoExceptionOccurred();
        thenResultingTensorListIsSize(1);
        thenResultingNamedTensorIsEqualTo("INPUT0", Float.class, Arrays.asList(1.0F));
    }

    @Test
    public void adapterShouldWorkWithDoubleScalarWiredRecord() {
        givenWireRecordPropWith("INPUT0", new DoubleValue(3.0D));
        givenWireRecord();

        givenTensorDescriptorWith("INPUT0", "FP32", Arrays.asList(1L, 1L));

        whenTensorListAdapterConvertsFromWireRecord();

        thenNoExceptionOccurred();
        thenResultingTensorListIsSize(1);
        thenResultingNamedTensorIsEqualTo("INPUT0", Double.class, Arrays.asList(3.0D));
    }

    @Test
    public void adapterShouldWorkWithIntegerScalarWiredRecord() {
        givenWireRecordPropWith("INPUT0", new IntegerValue(6));
        givenWireRecord();

        givenTensorDescriptorWith("INPUT0", "INT32", Arrays.asList(1L, 1L));

        whenTensorListAdapterConvertsFromWireRecord();

        thenNoExceptionOccurred();
        thenResultingTensorListIsSize(1);
        thenResultingNamedTensorIsEqualTo("INPUT0", Integer.class, Arrays.asList(6));
    }

    @Test
    public void adapterShouldWorkWithLongScalarWiredRecord() {
        givenWireRecordPropWith("INPUT0", new LongValue(6555L));
        givenWireRecord();

        givenTensorDescriptorWith("INPUT0", "INT32", Arrays.asList(1L, 1L));

        whenTensorListAdapterConvertsFromWireRecord();

        thenNoExceptionOccurred();
        thenResultingTensorListIsSize(1);
        thenResultingNamedTensorIsEqualTo("INPUT0", Long.class, Arrays.asList(6555L));
    }

    @Test
    public void adapterShouldWorkWithStringScalarWiredRecord() {
        givenWireRecordPropWith("INPUT0", new StringValue("This is a test"));
        givenWireRecord();

        givenTensorDescriptorWith("INPUT0", "STRING", Arrays.asList(1L, 1L));

        whenTensorListAdapterConvertsFromWireRecord();

        thenNoExceptionOccurred();
        thenResultingTensorListIsSize(1);
        thenResultingNamedTensorIsEqualTo("INPUT0", String.class, Arrays.asList("This is a test"));
    }

    @Test
    public void adapterShouldThrowIfChannelAndTensorNamesDontMatch() {
        givenWireRecordPropWith("INPUT0", new BooleanValue(true));
        givenWireRecord();

        givenTensorDescriptorWith("INPUT1", "BOOL", Arrays.asList(1L, 1L));

        whenTensorListAdapterConvertsFromWireRecord();

        thenExceptionOccurred();
    }

    @Test
    public void adapterShouldWorkWithMultipleBooleanWiredRecord() {
        givenWireRecordPropWith("INPUT0", new BooleanValue(true));
        givenWireRecordPropWith("INPUT1", new BooleanValue(false));
        givenWireRecordPropWith("INPUT3", new BooleanValue(true));
        givenWireRecord();

        givenTensorDescriptorWith("INPUT0", "BOOL", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("INPUT1", "BOOL", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("INPUT3", "BOOL", Arrays.asList(1L, 1L));

        whenTensorListAdapterConvertsFromWireRecord();

        thenNoExceptionOccurred();
        thenResultingTensorListIsSize(3);
        thenResultingNamedTensorIsEqualTo("INPUT0", Boolean.class, Arrays.asList(true));
        thenResultingNamedTensorIsEqualTo("INPUT1", Boolean.class, Arrays.asList(false));
        thenResultingNamedTensorIsEqualTo("INPUT3", Boolean.class, Arrays.asList(true));
    }

    @Test
    public void adapterShouldWorkWithMultipleByteArraysWiredRecord() {
        givenWireRecordPropWith("INPUT0", new ByteArrayValue(new byte[] { 1, 2, 3, 4 }));
        givenWireRecordPropWith("INPUT1", new ByteArrayValue(new byte[] { 3, 4 }));
        givenWireRecordPropWith("INPUT3", new ByteArrayValue(new byte[] { 1, 2, 3, 4, 5, 10 }));
        givenWireRecord();

        givenTensorDescriptorWith("INPUT0", "BYTES", Arrays.asList(1L, 4L));
        givenTensorDescriptorWith("INPUT1", "BYTES", Arrays.asList(1L, 2L));
        givenTensorDescriptorWith("INPUT3", "BYTES", Arrays.asList(1L, 6L));

        whenTensorListAdapterConvertsFromWireRecord();

        thenNoExceptionOccurred();
        thenResultingTensorListIsSize(3);
        thenResultingNamedTensorIsEqualTo("INPUT0", Byte.class, Arrays.asList((byte) 1, (byte) 2, (byte) 3, (byte) 4));
        thenResultingNamedTensorIsEqualTo("INPUT1", Byte.class, Arrays.asList((byte) 3, (byte) 4));
        thenResultingNamedTensorIsEqualTo("INPUT3", Byte.class,
                Arrays.asList((byte) 1, (byte) 2, (byte) 3, (byte) 4, (byte) 5, (byte) 10));
    }

    @Test
    public void adapterShouldWorkWithMultipleFloatWiredRecord() {
        givenWireRecordPropWith("INPUT0", new FloatValue(1.0F));
        givenWireRecordPropWith("INPUT1", new FloatValue(2.0F));
        givenWireRecordPropWith("INPUT2", new FloatValue(3.0F));
        givenWireRecordPropWith("INPUT3", new FloatValue(4.0F));
        givenWireRecord();

        givenTensorDescriptorWith("INPUT0", "FP32", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("INPUT1", "FP32", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("INPUT2", "FP32", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("INPUT3", "FP32", Arrays.asList(1L, 1L));

        whenTensorListAdapterConvertsFromWireRecord();

        thenNoExceptionOccurred();
        thenResultingTensorListIsSize(4);
        thenResultingNamedTensorIsEqualTo("INPUT0", Float.class, Arrays.asList(1.0F));
        thenResultingNamedTensorIsEqualTo("INPUT1", Float.class, Arrays.asList(2.0F));
        thenResultingNamedTensorIsEqualTo("INPUT2", Float.class, Arrays.asList(3.0F));
        thenResultingNamedTensorIsEqualTo("INPUT3", Float.class, Arrays.asList(4.0F));
    }

    @Test
    public void adapterShouldWorkWithMultipleDoubleWiredRecord() {
        givenWireRecordPropWith("INPUT2", new DoubleValue(3.0D));
        givenWireRecordPropWith("INPUT3", new DoubleValue(4.0D));
        givenWireRecord();

        givenTensorDescriptorWith("INPUT2", "FP32", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("INPUT3", "FP32", Arrays.asList(1L, 1L));

        whenTensorListAdapterConvertsFromWireRecord();

        thenNoExceptionOccurred();
        thenResultingTensorListIsSize(2);
        thenResultingNamedTensorIsEqualTo("INPUT2", Double.class, Arrays.asList(3.0D));
        thenResultingNamedTensorIsEqualTo("INPUT3", Double.class, Arrays.asList(4.0D));
    }

    @Test
    public void adapterShouldWorkWithMultipleIntegerWiredRecord() {
        givenWireRecordPropWith("INPUT2", new IntegerValue(30));
        givenWireRecordPropWith("INPUT3", new IntegerValue(42));
        givenWireRecord();

        givenTensorDescriptorWith("INPUT2", "INT32", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("INPUT3", "INT32", Arrays.asList(1L, 1L));

        whenTensorListAdapterConvertsFromWireRecord();

        thenNoExceptionOccurred();
        thenResultingTensorListIsSize(2);
        thenResultingNamedTensorIsEqualTo("INPUT2", Integer.class, Arrays.asList(30));
        thenResultingNamedTensorIsEqualTo("INPUT3", Integer.class, Arrays.asList(42));
    }

    @Test
    public void adapterShouldWorkWithMultipleLongWiredRecord() {
        givenWireRecordPropWith("INPUT2", new LongValue(30));
        givenWireRecordPropWith("INPUT3", new LongValue(42));
        givenWireRecord();

        givenTensorDescriptorWith("INPUT2", "INT32", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("INPUT3", "INT32", Arrays.asList(1L, 1L));

        whenTensorListAdapterConvertsFromWireRecord();

        thenNoExceptionOccurred();
        thenResultingTensorListIsSize(2);
        thenResultingNamedTensorIsEqualTo("INPUT2", Long.class, Arrays.asList(30L));
        thenResultingNamedTensorIsEqualTo("INPUT3", Long.class, Arrays.asList(42L));
    }

    @Test
    public void adapterShouldWorkWithMultipleStringsWiredRecord() {
        givenWireRecordPropWith("INPUT2", new StringValue("This is a test string"));
        givenWireRecordPropWith("INPUT3", new StringValue("This is another string for testing"));
        givenWireRecord();

        givenTensorDescriptorWith("INPUT2", "STRING", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("INPUT3", "STRING", Arrays.asList(1L, 1L));

        whenTensorListAdapterConvertsFromWireRecord();

        thenNoExceptionOccurred();
        thenResultingTensorListIsSize(2);
        thenResultingNamedTensorIsEqualTo("INPUT2", String.class, Arrays.asList("This is a test string"));
        thenResultingNamedTensorIsEqualTo("INPUT3", String.class, Arrays.asList("This is another string for testing"));
    }

    @Test
    public void adapterShouldWorkWithMultipleDifferentTypeWiredRecord() {
        givenWireRecordPropWith("INPUT0", new FloatValue(1.0F));
        givenWireRecordPropWith("INPUT1", new BooleanValue(true));
        givenWireRecordPropWith("INPUT2", new IntegerValue(64));
        givenWireRecordPropWith("INPUT3", new LongValue(65535));
        givenWireRecord();

        givenTensorDescriptorWith("INPUT0", "FP32", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("INPUT1", "BOOL", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("INPUT2", "INT32", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("INPUT3", "INT32", Arrays.asList(1L, 1L));

        whenTensorListAdapterConvertsFromWireRecord();

        thenNoExceptionOccurred();
        thenResultingTensorListIsSize(4);
        thenResultingNamedTensorIsEqualTo("INPUT0", Float.class, Arrays.asList(1.0F));
        thenResultingNamedTensorIsEqualTo("INPUT1", Boolean.class, Arrays.asList(true));
        thenResultingNamedTensorIsEqualTo("INPUT2", Integer.class, Arrays.asList(64));
        thenResultingNamedTensorIsEqualTo("INPUT3", Long.class, Arrays.asList(65535L));
    }

    @Test
    public void adapterShouldWorkWithBooleanScalarTensor() {
        givenTensorDescriptorWith("OUTPUT0", "BOOL", Arrays.asList(1L, 1L));
        givenTensorWith("OUTPUT0", "BOOL", Arrays.asList(1L, 1L), Boolean.class, Arrays.asList(true));

        whenTensorListAdapterConvertsFromTensorList();

        thenNoExceptionOccurred();
        thenResultingWireRecordIsSize(1);
        thenAllWireRecordsHaveSingleProperty();
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT0", new BooleanValue(true));
    }

    @Test
    public void adapterShouldWorkWithByteArrayScalarTensor() {
        givenTensorDescriptorWith("OUTPUT0", "BYTES", Arrays.asList(1L, 1L));
        givenTensorWith("OUTPUT0", "BYTES", Arrays.asList(1L, 1L), Byte.class,
                Arrays.asList((byte) 1, (byte) 2, (byte) 3));

        whenTensorListAdapterConvertsFromTensorList();

        thenNoExceptionOccurred();
        thenResultingWireRecordIsSize(1);
        thenAllWireRecordsHaveSingleProperty();
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT0", new ByteArrayValue(new byte[] { 1, 2, 3 }));
    }

    @Test
    public void adapterShouldWorkWithFloatScalarTensor() {
        givenTensorDescriptorWith("OUTPUT0", "FP32", Arrays.asList(1L, 1L));
        givenTensorWith("OUTPUT0", "FP32", Arrays.asList(1L, 1L), Float.class, Arrays.asList(3.2F));

        whenTensorListAdapterConvertsFromTensorList();

        thenNoExceptionOccurred();
        thenResultingWireRecordIsSize(1);
        thenAllWireRecordsHaveSingleProperty();
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT0", new FloatValue(3.2F));
    }

    @Test
    public void adapterShouldWorkWithDoubleScalarTensor() {
        givenTensorDescriptorWith("OUTPUT0", "FP32", Arrays.asList(1L, 1L));
        givenTensorWith("OUTPUT0", "FP32", Arrays.asList(1L, 1L), Double.class, Arrays.asList(5.464D));

        whenTensorListAdapterConvertsFromTensorList();

        thenNoExceptionOccurred();
        thenResultingWireRecordIsSize(1);
        thenAllWireRecordsHaveSingleProperty();
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT0", new DoubleValue(5.464D));
    }

    @Test
    public void adapterShouldWorkWithIntegerScalarTensor() {
        givenTensorDescriptorWith("OUTPUT0", "INT32", Arrays.asList(1L, 1L));
        givenTensorWith("OUTPUT0", "INT32", Arrays.asList(1L, 1L), Integer.class, Arrays.asList(42));

        whenTensorListAdapterConvertsFromTensorList();

        thenNoExceptionOccurred();
        thenResultingWireRecordIsSize(1);
        thenAllWireRecordsHaveSingleProperty();
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT0", new IntegerValue(42));
    }

    @Test
    public void adapterShouldWorkWithLongScalarTensor() {
        givenTensorDescriptorWith("OUTPUT0", "INT32", Arrays.asList(1L, 1L));
        givenTensorWith("OUTPUT0", "INT32", Arrays.asList(1L, 1L), Long.class, Arrays.asList(36L));

        whenTensorListAdapterConvertsFromTensorList();

        thenNoExceptionOccurred();
        thenResultingWireRecordIsSize(1);
        thenAllWireRecordsHaveSingleProperty();
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT0", new LongValue(36L));
    }

    @Test
    public void adapterShouldWorkWithStringScalarTensor() {
        givenTensorDescriptorWith("OUTPUT0", "STRING", Arrays.asList(1L, 1L));
        givenTensorWith("OUTPUT0", "STRING", Arrays.asList(1L, 1L), String.class,
                Arrays.asList("This is a test string"));

        whenTensorListAdapterConvertsFromTensorList();

        thenNoExceptionOccurred();
        thenResultingWireRecordIsSize(1);
        thenAllWireRecordsHaveSingleProperty();
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT0", new StringValue("This is a test string"));
    }

    @Test
    public void adapterShouldWorkWithMultipleBooleanTensor() {
        givenTensorDescriptorWith("OUTPUT0", "BOOL", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("OUTPUT1", "BOOL", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("OUTPUT2", "BOOL", Arrays.asList(1L, 1L));

        givenTensorWith("OUTPUT0", "BOOL", Arrays.asList(1L, 1L), Boolean.class, Arrays.asList(true));
        givenTensorWith("OUTPUT1", "BOOL", Arrays.asList(1L, 1L), Boolean.class, Arrays.asList(false));
        givenTensorWith("OUTPUT2", "BOOL", Arrays.asList(1L, 1L), Boolean.class, Arrays.asList(true));

        whenTensorListAdapterConvertsFromTensorList();

        thenNoExceptionOccurred();
        thenResultingWireRecordIsSize(3);
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT0", new BooleanValue(true));
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT1", new BooleanValue(false));
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT2", new BooleanValue(true));
    }

    @Test
    public void adapterShouldWorkWithMultipleByteArrayTensor() {
        givenTensorDescriptorWith("OUTPUT0", "BYTES", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("OUTPUT1", "BYTES", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("OUTPUT2", "BYTES", Arrays.asList(1L, 1L));

        givenTensorWith("OUTPUT0", "BYTES", Arrays.asList(1L, 1L), Byte.class,
                Arrays.asList((byte) 1, (byte) 2, (byte) 3));
        givenTensorWith("OUTPUT1", "BYTES", Arrays.asList(1L, 1L), Byte.class, Arrays.asList((byte) 16, (byte) 5));
        givenTensorWith("OUTPUT2", "BYTES", Arrays.asList(1L, 1L), Byte.class, Arrays.asList((byte) 1));

        whenTensorListAdapterConvertsFromTensorList();

        thenNoExceptionOccurred();
        thenResultingWireRecordIsSize(3);
        thenAllWireRecordsHaveSingleProperty();
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT0", new ByteArrayValue(new byte[] { 1, 2, 3 }));
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT1", new ByteArrayValue(new byte[] { 16, 5 }));
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT2", new ByteArrayValue(new byte[] { 1 }));
    }

    @Test
    public void adapterShouldWorkWithMultipleFloatTensor() {
        givenTensorDescriptorWith("OUTPUT0", "FP32", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("OUTPUT1", "FP32", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("OUTPUT2", "FP32", Arrays.asList(1L, 1L));

        givenTensorWith("OUTPUT0", "FP32", Arrays.asList(1L, 1L), Float.class, Arrays.asList(3.2F));
        givenTensorWith("OUTPUT1", "FP32", Arrays.asList(1L, 1L), Float.class, Arrays.asList(55.66F));
        givenTensorWith("OUTPUT2", "FP32", Arrays.asList(1L, 1L), Float.class, Arrays.asList(-12.5F));

        whenTensorListAdapterConvertsFromTensorList();

        thenNoExceptionOccurred();
        thenResultingWireRecordIsSize(3);
        thenAllWireRecordsHaveSingleProperty();
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT0", new FloatValue(3.2F));
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT1", new FloatValue(55.66F));
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT2", new FloatValue(-12.5F));
    }

    @Test
    public void adapterShouldWorkWithMultipleDoubleTensor() {
        givenTensorDescriptorWith("OUTPUT1", "FP32", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("OUTPUT2", "FP32", Arrays.asList(1L, 1L));

        givenTensorWith("OUTPUT1", "FP32", Arrays.asList(1L, 1L), Double.class, Arrays.asList(55.66D));
        givenTensorWith("OUTPUT2", "FP32", Arrays.asList(1L, 1L), Double.class, Arrays.asList(-12.5D));

        whenTensorListAdapterConvertsFromTensorList();

        thenNoExceptionOccurred();
        thenResultingWireRecordIsSize(2);
        thenAllWireRecordsHaveSingleProperty();
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT1", new DoubleValue(55.66D));
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT2", new DoubleValue(-12.5D));
    }

    @Test
    public void adapterShouldWorkWithMultipleIntegerTensor() {
        givenTensorDescriptorWith("OUTPUT0", "INT32", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("OUTPUT1", "INT32", Arrays.asList(1L, 1L));

        givenTensorWith("OUTPUT0", "INT32", Arrays.asList(1L, 1L), Integer.class, Arrays.asList(35));
        givenTensorWith("OUTPUT1", "INT32", Arrays.asList(1L, 1L), Integer.class, Arrays.asList(55));

        whenTensorListAdapterConvertsFromTensorList();

        thenNoExceptionOccurred();
        thenResultingWireRecordIsSize(2);
        thenAllWireRecordsHaveSingleProperty();
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT0", new IntegerValue(35));
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT1", new IntegerValue(55));
    }

    @Test
    public void adapterShouldWorkWithMultipleLongTensor() {
        givenTensorDescriptorWith("OUTPUT0", "INT32", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("OUTPUT1", "INT32", Arrays.asList(1L, 1L));

        givenTensorWith("OUTPUT0", "INT32", Arrays.asList(1L, 1L), Long.class, Arrays.asList(356L));
        givenTensorWith("OUTPUT1", "INT32", Arrays.asList(1L, 1L), Long.class, Arrays.asList(-55L));

        whenTensorListAdapterConvertsFromTensorList();

        thenNoExceptionOccurred();
        thenResultingWireRecordIsSize(2);
        thenAllWireRecordsHaveSingleProperty();
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT0", new LongValue(356L));
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT1", new LongValue(-55L));
    }

    @Test
    public void adapterShouldWorkWithMultipleStringTensor() {
        givenTensorDescriptorWith("OUTPUT0", "STRING", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("OUTPUT1", "STRING", Arrays.asList(1L, 1L));

        givenTensorWith("OUTPUT0", "STRING", Arrays.asList(1L, 1L), String.class, Arrays.asList("This is a"));
        givenTensorWith("OUTPUT1", "STRING", Arrays.asList(1L, 1L), String.class, Arrays.asList("test string"));

        whenTensorListAdapterConvertsFromTensorList();

        thenNoExceptionOccurred();
        thenResultingWireRecordIsSize(2);
        thenAllWireRecordsHaveSingleProperty();
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT0", new StringValue("This is a"));
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT1", new StringValue("test string"));
    }

    @Test
    public void adapterShouldWorkWithMultipleDifferentTypeTensor() {
        givenTensorDescriptorWith("OUTPUT0", "FP32", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("OUTPUT1", "INT32", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("OUTPUT2", "STRING", Arrays.asList(1L, 1L));
        givenTensorDescriptorWith("OUTPUT3", "INT32", Arrays.asList(1L, 1L));

        givenTensorWith("OUTPUT0", "FP32", Arrays.asList(1L, 1L), Float.class, Arrays.asList(6.9F));
        givenTensorWith("OUTPUT1", "INT32", Arrays.asList(1L, 1L), Integer.class, Arrays.asList(100));
        givenTensorWith("OUTPUT2", "STRING", Arrays.asList(1L, 1L), String.class,
                Arrays.asList("May the force be with you"));
        givenTensorWith("OUTPUT3", "INT32", Arrays.asList(1L, 1L), Long.class, Arrays.asList(254678L));

        whenTensorListAdapterConvertsFromTensorList();

        thenNoExceptionOccurred();
        thenResultingWireRecordIsSize(4);
        thenAllWireRecordsHaveSingleProperty();
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT0", new FloatValue(6.9F));
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT1", new IntegerValue(100));
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT2", new StringValue("May the force be with you"));
        thenResultingNamedWireRecordPropertiesAreEqualTo("OUTPUT3", new LongValue(254678L));
    }

    /*
     * Given
     */
    private void givenWireRecordPropWith(String name, TypedValue<?> value) {
        this.wireRecordProperties.put(name, value);
    }

    private void givenWireRecord() {
        this.inputRecord = new WireRecord(this.wireRecordProperties);
    }

    private void givenTensorDescriptorWith(String name, String type, List<Long> shape) {
        Optional<String> format = Optional.empty();
        Map<String, Object> parameters = new HashMap<>();

        TensorDescriptor descriptor = new TensorDescriptor(name, type, format, shape, parameters);

        this.inputDescriptors.add(descriptor);
    }

    private <T> void givenTensorWith(String name, String type, List<Long> shape, Class<T> classType, List<T> data) {
        Optional<String> format = Optional.empty();
        Map<String, Object> parameters = new HashMap<>();

        TensorDescriptor descriptor = new TensorDescriptor(name, type, format, shape, parameters);

        Tensor tensor = new Tensor(classType, descriptor, data);

        this.inputTensors.add(tensor);
    }

    /*
     * When
     */
    private void whenTensorListAdapterConvertsFromWireRecord() {
        try {
            this.outputTensors = TensorListAdapter.givenDescriptors(this.inputDescriptors).fromWireRecord(inputRecord);
        } catch (KuraException e) {
            e.printStackTrace();
            this.exceptionOccurred = true;
        }
    }

    private void whenTensorListAdapterConvertsFromTensorList() {
        try {
            this.outputRecords = TensorListAdapter.givenDescriptors(this.inputDescriptors).fromTensorList(inputTensors);
        } catch (KuraIOException e) {
            e.printStackTrace();
            this.exceptionOccurred = true;
        }
    }

    /*
     * Then
     */
    private void thenNoExceptionOccurred() {
        assertFalse(this.exceptionOccurred);
    }

    private void thenExceptionOccurred() {
        assertTrue(this.exceptionOccurred);
    }

    private void thenResultingTensorListIsSize(int size) {
        assertFalse(this.outputTensors.isEmpty());
        assertEquals(size, this.outputTensors.size());
    }

    private void thenResultingWireRecordIsSize(int size) {
        assertFalse(this.outputRecords.isEmpty());
        assertEquals(size, this.outputRecords.size());
    }

    private void thenAllWireRecordsHaveSingleProperty() {
        assertFalse(this.outputRecords.isEmpty());

        for (WireRecord record : outputRecords) {
            Map<String, TypedValue<?>> properties = record.getProperties();

            assertEquals(1, properties.size());
        }
    }

    private <T> void thenResultingNamedTensorIsEqualTo(String name, Class<T> type, List<T> expectedData) {
        Tensor tensor = findTensorByName(name, outputTensors);

        assertNotNull(tensor);

        Optional<List<T>> data = tensor.getData(type);

        assertTrue(data.isPresent());
        assertEquals(expectedData, data.get());
    }

    private Tensor findTensorByName(String name, List<Tensor> tensorList) {
        for (Tensor currTensor : tensorList) {
            String currTensorName = currTensor.getDescriptor().getName();

            if (currTensorName.equals(name)) {
                return currTensor;
            }
        }

        return null;
    }

    private void thenResultingNamedWireRecordPropertiesAreEqualTo(String channelName, TypedValue<?> data) {
        TypedValue<?> value = findWireRecordPropByChannelName(channelName, this.outputRecords);

        assertNotNull(value);
        assertEquals(data, value);
    }

    private TypedValue<?> findWireRecordPropByChannelName(String channelName, List<WireRecord> records) {
        for (WireRecord record : records) {
            Map<String, TypedValue<?>> properties = record.getProperties();

            if (properties.containsKey(channelName)) {
                return properties.get(channelName);
            }
        }

        return null;
    }

    /*
     * Utils
     */
    @Before
    public void cleanup() {
        this.wireRecordProperties = new HashMap<String, TypedValue<?>>();
        this.inputDescriptors = new ArrayList<>();
        this.inputTensors = new ArrayList<>();
    }

}
